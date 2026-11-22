"""Service errors, mapped to HTTP status codes by the API."""

from __future__ import annotations


class ServiceError(Exception):
    status_code = 400


class NotFound(ServiceError):
    status_code = 404


class Conflict(ServiceError):
    status_code = 409


class InvalidRequest(ServiceError):
    status_code = 422
