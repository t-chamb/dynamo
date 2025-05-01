# Interface Example

This example demonstrates how to use `DynamoServiceInterface` and `abstract_dynamo_endpoint` to define and implement service interfaces in Dynamo, following the SDK constraints.

## Key Concepts

1. **Service Interface**: A class that inherits from `DynamoServiceInterface` and defines abstract methods using `@abstract_dynamo_endpoint`
2. **Concrete Implementation**: A class that inherits from the interface and implements all abstract methods with `@dynamo_endpoint`
3. **Service Validation**: The `@service` decorator ensures that all abstract methods are properly implemented
4. **SDK Constraints**:
   - Endpoints must be async generators
   - Endpoints must accept a single Pydantic model argument
   - Services must expose HTTP ingress
   - Endpoints must be decorated with `is_api=True` to be exposed via HTTP

## Example Structure

The example implements a calculator service with four operations:
- Addition
- Subtraction
- Multiplication
- Division

### Interface Definition

```python
class CalculatorInterface(DynamoServiceInterface):
    @abstract_dynamo_endpoint
    async def calculate(self, request: CalculatorRequest) -> AsyncGenerator[CalculatorResponse, None]:
        """Perform a calculator operation."""
        pass
```

### Concrete Implementation

```python
@service(
    dynamo={
        "enabled": True,
        "namespace": "calculator",
    },
    image=DYNAMO_IMAGE,
    app=app,
)
class CalculatorService(CalculatorInterface):
    @dynamo_endpoint(is_api=True)
    async def calculate(self, request: CalculatorRequest) -> AsyncGenerator[CalculatorResponse, None]:
        operation = request.operation.lower()
        a, b = request.a, request.b
        # ... perform operation ...
        yield CalculatorResponse(result=result, operation=operation)
```

## Running the Example

```bash
python service.py
```

The service will start and be available at http://localhost:8000. You can test it with curl:

```bash
# Addition
curl -X POST http://localhost:8000/calculate \
  -H "Content-Type: application/json" \
  -d '{"operation": "add", "a": 1, "b": 2}'

# Subtraction
curl -X POST http://localhost:8000/calculate \
  -H "Content-Type: application/json" \
  -d '{"operation": "subtract", "a": 5, "b": 2}'

# Multiplication
curl -X POST http://localhost:8000/calculate \
  -H "Content-Type: application/json" \
  -d '{"operation": "multiply", "a": 3, "b": 4}'

# Division
curl -X POST http://localhost:8000/calculate \
  -H "Content-Type: application/json" \
  -d '{"operation": "divide", "a": 10, "b": 2}'
```

## Key Points

1. All interface methods must be implemented in the concrete class
2. All implemented methods must be decorated with `@dynamo_endpoint`
3. The concrete class must be decorated with `@service`
4. Endpoints must be async generators
5. Endpoints must accept a single Pydantic model argument
6. Endpoints must be decorated with `is_api=True` to be exposed via HTTP
7. The interface cannot be instantiated directly 