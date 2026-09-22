# Production image for the gateway — a standalone reverse proxy that authorizes A2A
# and MCP traffic in front of your agents/servers.
#
#   docker build -f examples/gateway/Dockerfile -t axio-gateway .
#
# Routes are DATA, supplied at run time (not baked into the image), so one image fronts any
# set of servers. Provide them as a mounted JSON file (recommended) or inline env JSON:
#
#   # routes.json — a list of route objects
#   [
#     {"path_prefix":"/billing","target_url":"http://host.docker.internal:9211","protocol":"a2a","agent_name":"billing-agent"},
#     {"path_prefix":"/tools","target_url":"http://host.docker.internal:9212/mcp","protocol":"mcp"}
#   ]
#
#   docker run -p 9210:9210 \
#     -e PDP_BASE_URL=http://host.docker.internal:8080 \
#     -e GATEWAY_ROUTES_FILE=/config/routes.json \
#     -v "$PWD/routes.json:/config/routes.json:ro" \
#     axio-gateway
#
# (host.docker.internal reaches host services from the container on Docker Desktop; in a
# real deployment these are your service DNS names.)
#
# TRUST BOUNDARY: the gateway authorizes, it does not authenticate. The default context
# provider reads JWT claims without verifying the signature, so run it behind an ingress
# that authenticates and terminates TLS. If this gateway IS your edge, supply a
# context_provider that overrides get_subject to verify — see the gateway README.
FROM python:3.14-slim

WORKDIR /src
COPY . /src

# Install the SDK with the gateway extra. fastmcp pulls in uvicorn transitively; we also
# name it explicitly since the CMD runs the app with it. (build_gateway() just returns an
# ASGI app — the server is a deployment choice, so uvicorn lives here, not in the extra.)
RUN pip install --no-cache-dir "/src[gateway]" "uvicorn==0.44.0"

# `examples.gateway.gateway:app` builds the app from GATEWAY_ROUTES_FILE / GATEWAY_ROUTES.
# Point the CMD at your own module for production. PYTHONPATH=/src makes examples importable.
ENV PYTHONPATH=/src PDP_BASE_URL=http://localhost:8080

# Run as an unprivileged user. The gateway needs no write access to /src and binds an
# unprivileged port, so there is nothing for root to do here except widen the blast radius
# of a container escape.
RUN useradd --create-home --uid 10001 gateway
USER 10001

EXPOSE 9210

# Liveness for orchestrators. The gateway mounts only the routes you configure — there is no
# unauthenticated /health endpoint to poll — so this checks that uvicorn is accepting
# connections on the listen port, which is the strongest signal available without credentials.
# A failed connection raises, and an uncaught exception exits non-zero, which is the unhealthy signal.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import socket; socket.create_connection(('127.0.0.1', 9210), 2).close()"]

CMD ["uvicorn", "examples.gateway.gateway:app", "--host", "0.0.0.0", "--port", "9210"]
