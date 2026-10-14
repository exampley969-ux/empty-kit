"""
Static checks of the FastAPI adapter that run WITHOUT fastapi installed.

The adapter can't be executed in an environment lacking fastapi, so these
tests read its source and verify two things that would otherwise only show
up at runtime:
  1. every `service.<method>(...)` call matches a real ApiService method
     with a compatible signature (no typos, no missing/extra arguments);
  2. every route except login takes the `token` dependency - i.e. no
     endpoint was accidentally left unauthenticated (Section 28).
"""
import ast
import inspect
import os
import unittest

from vision_radar.api.service import ApiService

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "api", "fastapi_app.py")
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def parse():
    with open(APP_PATH, encoding="utf-8") as f:
        return ast.parse(f.read())


def service_calls(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "service"):
            yield node


def route_functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                        and isinstance(dec.func.value, ast.Name) and dec.func.value.id == "app"
                        and dec.func.attr in HTTP_METHODS):
                    yield node, dec


class TestAdapterMatchesService(unittest.TestCase):
    def test_every_service_call_matches_a_real_method_signature(self):
        calls = list(service_calls(parse()))
        self.assertGreater(len(calls), 15)                    # sanity: we really found the routes
        for call in calls:
            name = call.func.attr
            self.assertTrue(hasattr(ApiService, name), f"service.{name} does not exist")
            params = list(inspect.signature(getattr(ApiService, name)).parameters.values())[1:]   # drop self
            args = list(call.args)
            if params and params[0].name == "user":            # decorated endpoint: adapter passes the token first
                params = params[1:]
                self.assertTrue(args, f"service.{name}() is missing its token argument")
                self.assertIsInstance(args[0], ast.Name, f"service.{name}: first argument should be `token`")
                self.assertEqual(args[0].id, "token", f"service.{name}: first argument should be `token`")
                args = args[1:]
            names = [p.name for p in params]
            self.assertLessEqual(len(args), len(names), f"service.{name}: too many positional arguments")
            kw = [k.arg for k in call.keywords]
            for k in kw:
                self.assertIn(k, names, f"service.{name}: unknown keyword '{k}'")
            provided = set(names[:len(args)]) | set(kw)
            required = {p.name for p in params if p.default is inspect.Parameter.empty}
            self.assertLessEqual(required, provided, f"service.{name}: missing {required - provided}")

    def test_every_public_service_endpoint_is_exposed_by_the_adapter(self):
        exposed = {c.func.attr for c in service_calls(parse())}
        internal = {"shutdown"}
        public = {n for n, m in inspect.getmembers(ApiService, inspect.isfunction)
                  if not n.startswith("_") and n not in internal}
        self.assertEqual(public - exposed, set(), "ApiService endpoints missing from the HTTP adapter")


class TestNoUnauthenticatedRoutes(unittest.TestCase):
    def test_every_route_except_login_requires_the_token_dependency(self):
        routes = list(route_functions(parse()))
        self.assertGreater(len(routes), 15)
        for func, dec in routes:
            argnames = [a.arg for a in func.args.args]
            if func.name == "login":
                continue
            self.assertIn("token", argnames, f"route {func.name} has no auth dependency")

    def test_login_is_the_only_route_without_a_token_parameter(self):
        without = [f.name for f, _ in route_functions(parse()) if "token" not in [a.arg for a in f.args.args]]
        self.assertEqual(without, ["login"])


if __name__ == "__main__":
    unittest.main()
