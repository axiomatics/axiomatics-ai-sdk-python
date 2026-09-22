# Security policy

## Reporting a vulnerability

Please do not report security issues in a public issue or pull request. Use GitHub's private
vulnerability reporting instead: on the repository page open **Security → Report a
vulnerability**. Include the SDK version, the integration path affected, and steps to reproduce.
You will receive an acknowledgement within five business days.

## Scope

This SDK is a Policy Enforcement Point: it builds authorization requests and applies the PDP's
decisions. Reports about incorrect enforcement, decision correlation, or anything that lets a
denied action through are in scope. The trust assumptions the SDK relies on are documented in
[docs/security.md](docs/security.md); the most important one is that the SDK authorizes but does
not authenticate, so an unverified bearer token is treated as a deployment precondition, not a
vulnerability.

## Supported versions

The latest released minor version receives fixes.
