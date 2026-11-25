"""AI/ML API as a named generation provider: priority, fixed host, no key leaks."""
import http.server
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import builtin
import generate
import settings_config
import userconfig


def _providers(**keys):
    """userconfig.provider stub: keys like OPENAI=('k', 'base', 'model')."""
    def provider(prefix):
        key, base, model = keys.get(prefix, ("", "", ""))
        return {"key": key, "base": base, "model": model,
                "source": "test" if key else "none"}
    return provider


class CredentialPriorityTests(unittest.TestCase):
    def setUp(self):
        generate._BUILTIN_MODEL = None

    def test_existing_openai_key_is_untouched_by_aimlapi_key(self):
        stub = _providers(OPENAI=("own", "https://api.deepseek.com", "deepseek-chat"),
                          AIMLAPI=("aiml", "", ""))
        with patch.object(userconfig, "provider", side_effect=stub):
            base, key, model, _src, api = generate.load_credentials()
        self.assertEqual((base, key, model, api),
                         ("https://api.deepseek.com", "own", "deepseek-chat", "openai"))

    def test_existing_anthropic_key_is_untouched_by_aimlapi_key(self):
        stub = _providers(ANTHROPIC=("anth", "https://open.bigmodel.cn/api/anthropic", "glm-4-flash"),
                          AIMLAPI=("aiml", "", ""))
        with patch.object(userconfig, "provider", side_effect=stub):
            base, key, _model, _src, api = generate.load_credentials()
        self.assertEqual((base, key, api),
                         ("https://open.bigmodel.cn/api/anthropic", "anth", "anthropic"))

    def test_aimlapi_key_alone_selects_aimlapi(self):
        with patch.object(userconfig, "provider", side_effect=_providers(AIMLAPI=("aiml", "", ""))):
            base, key, model, src, api = generate.load_credentials()
        self.assertEqual((base, key, model, src, api),
                         ("https://api.aimlapi.com/v1", "aiml", "deepseek/deepseek-chat",
                          "test", "openai"))
        self.assertEqual(generate._endpoint(base, api),
                         "https://api.aimlapi.com/v1/chat/completions")

    def test_aimlapi_model_is_configurable(self):
        stub = _providers(AIMLAPI=("aiml", "", "alibaba/qwen3.8-flash"))
        with patch.object(userconfig, "provider", side_effect=stub):
            self.assertEqual(generate.load_credentials()[2], "alibaba/qwen3.8-flash")

    def test_aimlapi_key_never_follows_a_foreign_base_url(self):
        for foreign in ("https://evil.example/v1", "https://api.aimlapi.com.evil.example/v1",
                        "http://api.aimlapi.com/v1", "https://xapi.aimlapi.com/v1"):
            with self.subTest(base=foreign), patch.object(
                    userconfig, "provider", side_effect=_providers(AIMLAPI=("aiml", foreign, ""))):
                self.assertEqual(generate.load_credentials()[0], "https://api.aimlapi.com/v1")

    def test_no_keys_keeps_builtin_then_none(self):
        with patch.object(userconfig, "provider", side_effect=_providers()), \
             patch.object(builtin, "API_KEY", "shared"), \
             patch.object(generate.urllib.request, "urlopen", side_effect=OSError):
            self.assertEqual(generate.load_credentials()[3], generate.BUILTIN_SOURCE)
        with patch.object(userconfig, "provider", side_effect=_providers()), \
             patch.object(builtin, "API_KEY", ""):
            self.assertEqual(generate.load_credentials()[1], "")


class HostCheckTests(unittest.TestCase):
    def test_exact_host_only(self):
        self.assertTrue(generate.is_aimlapi_url("https://api.aimlapi.com/v1/chat/completions"))
        self.assertTrue(generate.is_aimlapi_url("https://API.aimlapi.com/v1"))
        for url in ("https://api.aimlapi.com.evil.example/v1", "https://evil.example/api.aimlapi.com",
                    "http://api.aimlapi.com/v1", "https://aimlapi.com/v1", ""):
            with self.subTest(url=url):
                self.assertFalse(generate.is_aimlapi_url(url))

    def test_settings_reject_foreign_aimlapi_endpoint_before_sending_key(self):
        with patch.object(settings_config.http.client, "HTTPSConnection",
                          side_effect=AssertionError("must not connect")), \
             patch.object(settings_config, "http_post_json",
                          side_effect=AssertionError("must not post")):
            with self.assertRaises(ValueError):
                settings_config.list_models("AIMLAPI", "https://evil.example/v1", "k")
            with self.assertRaises(ValueError):
                settings_config.test_connection("AIMLAPI", "https://evil.example/v1", "k", "m")
        # other services keep accepting any endpoint
        self.assertEqual(settings_config.validate_service_endpoint("OPENAI", "https://evil.example/v1"),
                         "https://evil.example/v1")

    def test_settings_expose_aimlapi_service(self):
        self.assertIn("AIMLAPI", settings_config.PREFIXES)
        self.assertEqual(settings_config.DEFAULTS["AIMLAPI"],
                         ("https://api.aimlapi.com/v1", "deepseek/deepseek-chat"))


class _Target(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        type(self).seen.append(self.headers.get("authorization"))
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"choices":[{"message":{"content":"ok"}}]}')

    def log_message(self, *args):
        pass


class _Redirector(http.server.BaseHTTPRequestHandler):
    location = ""

    def do_POST(self):
        self.send_response(307)
        self.send_header("location", self.location)
        self.end_headers()

    def log_message(self, *args):
        pass


def _serve(handler):
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class RedirectTests(unittest.TestCase):
    def test_stream_request_to_aimlapi_refuses_redirect_and_keeps_key(self):
        _Target.seen = []
        target = _serve(_Target)
        _Redirector.location = f"http://127.0.0.1:{target.server_port}/steal"
        redirector = _serve(_Redirector)
        self.addCleanup(target.server_close)
        self.addCleanup(target.shutdown)
        self.addCleanup(redirector.server_close)
        self.addCleanup(redirector.shutdown)
        local = f"http://127.0.0.1:{redirector.server_port}/v1/chat/completions"
        req = urllib.request.Request(local, data=b"{}",
                                     headers={"authorization": "Bearer secret"})
        # treat the local redirector as the AI/ML API host
        with patch.object(generate, "is_aimlapi_url", return_value=True):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                generate._open_stream(req, 5)
        self.assertEqual(ctx.exception.code, 307)
        self.assertEqual(_Target.seen, [])

    def test_generator_stream_uses_redirect_refusing_opener_for_aimlapi(self):
        gen = generate.Generator()
        stub = _providers(AIMLAPI=("aiml", "", ""))
        with patch.object(userconfig, "provider", side_effect=stub), \
             patch.object(generate._NO_REDIRECT, "open",
                          side_effect=urllib.error.URLError("blocked")) as opener, \
             patch.object(generate.urllib.request, "urlopen",
                          side_effect=AssertionError("default opener follows redirects")):
            with self.assertRaises(urllib.error.URLError):
                gen._call("hi", on_delta=lambda _f: None)
        sent = opener.call_args.args[0]
        self.assertEqual(sent.full_url, "https://api.aimlapi.com/v1/chat/completions")
        self.assertEqual(sent.get_header("Authorization"), "Bearer aiml")

    def test_other_providers_keep_default_opener(self):
        req = urllib.request.Request("https://api.deepseek.com/v1/chat/completions", data=b"{}")
        with patch.object(generate.urllib.request, "urlopen", return_value="default") as op, \
             patch.object(generate._NO_REDIRECT, "open", side_effect=AssertionError):
            self.assertEqual(generate._open_stream(req, 5), "default")
        op.assert_called_once()


if __name__ == "__main__":
    unittest.main()
