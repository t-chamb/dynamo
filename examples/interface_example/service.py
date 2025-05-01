"""
Interface Example
================

This example demonstrates how to use DynamoServiceInterface and abstract_dynamo_endpoint
to define and implement service interfaces with proper SDK constraints.

Run: python service.py

Expected output:
    Starting calculator service...
    Calculator service running on http://localhost:8000
    Try: curl -X POST http://localhost:8000/calculate -H "Content-Type: application/json" -d '{"operation": "add", "a": 1, "b": 2}'
"""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import DYNAMO_IMAGE, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


class CalculatorRequest(BaseModel):
    """Request model for calculator operations."""
    operation: str
    a: int
    b: int


class BinaryOperationRequest(BaseModel):
    """Request model for binary arithmetic operations."""
    a: int
    b: int


class CalculatorInterface(DynamoServiceInterface):
    """Interface defining calculator operations."""

    @abstract_dynamo_endpoint
    async def calculate(self, request: CalculatorRequest):
        """Perform a calculator operation."""
        pass

    @abstract_dynamo_endpoint
    async def add(self, request: BinaryOperationRequest):
        """Add two numbers."""
        pass

    @abstract_dynamo_endpoint
    async def subtract(self, request: BinaryOperationRequest):
        """Subtract b from a."""
        pass

    @abstract_dynamo_endpoint
    async def multiply(self, request: BinaryOperationRequest):
        """Multiply two numbers."""
        pass

    @abstract_dynamo_endpoint
    async def divide(self, request: BinaryOperationRequest):
        """Divide a by b."""
        pass


app = FastAPI(title="Calculator Service")


@service(
    dynamo={
        "enabled": True,
        "namespace": "calculator",
    },
    image=DYNAMO_IMAGE,
    app=app,
)
class CalculatorService(CalculatorInterface):
    """Concrete implementation of the calculator interface."""

    def __init__(self) -> None:
        # Configure logging
        configure_dynamo_logging(service_name="CalculatorService")
        logger.info("Starting calculator service")
        config = ServiceConfig.get_instance()
        self.port = config.get("CalculatorService", {}).get("port", 8000)
        logger.info(f"Calculator service port: {self.port}")

    @dynamo_endpoint(is_api=True)
    async def calculate(self, request: CalculatorRequest):
        """Perform a calculator operation."""
        operation = request.operation.lower()
        # Why doesn't dep take care of this for us?
        op_request = BinaryOperationRequest(a=request.a, b=request.b).model_dump_json()

        if operation == "add":
            async for result in self.add(op_request):
                yield result
        elif operation == "subtract":
            async for result in self.subtract(op_request):
                yield result
        elif operation == "multiply":
            async for result in self.multiply(op_request):
                yield result
        elif operation == "divide":
            async for result in self.divide(op_request):
                yield result
        else:
            raise ValueError(f"Unknown operation: {operation}")

    @dynamo_endpoint
    async def add(self, request: BinaryOperationRequest):
        """Add two numbers."""
        result = request.a + request.b
        yield f"Add: {request.a} + {request.b} = {result}"

    @dynamo_endpoint
    async def subtract(self, request: BinaryOperationRequest):
        """Subtract b from a."""
        result = request.a - request.b
        yield f"Subtract: {request.a} - {request.b} = {result}"

    @dynamo_endpoint
    async def multiply(self, request: BinaryOperationRequest):
        """Multiply two numbers."""
        result = request.a * request.b
        yield f"Multiply: {request.a} * {request.b} = {result}"

    @dynamo_endpoint
    async def divide(self, request: BinaryOperationRequest):
        """Divide a by b."""
        result = request.a / request.b
        yield f"Divide: {request.a} / {request.b} = {result}"
