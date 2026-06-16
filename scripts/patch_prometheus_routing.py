"""Patch prometheus-fastapi-instrumentator for Starlette IncludedRouter."""
from pathlib import Path

p = Path("/root/shuten-venv/lib/python3.11/site-packages/prometheus_fastapi_instrumentator/routing.py")
text = p.read_text()
text = text.replace("route_name = route.path", 'route_name = getattr(route, "path", None)')
p.write_text(text)
print("patched", p)
