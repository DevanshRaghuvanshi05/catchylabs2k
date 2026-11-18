# CatchyLabs Tests

This directory contains unit tests for the CatchyLabs research pipeline.

## Running Tests

Run all tests in the tests directory:

```bash
python -m unittest discover tests
```

Run a specific test file:

```bash
python tests/test_content_validation.py
```

Run a specific test class:

```bash
python -m unittest tests.test_content_validation.TestBrandClaimValidation
```

Run a specific test method:

```bash
python -m unittest tests.test_content_validation.TestBrandClaimValidation.test_brand_side_claim_rejected
```

## Test Coverage

- `test_content_validation.py`: Tests for Content Generation validation rules
  - Brand claim validation (product-scoped checking)
  - Voiceover length validation
  - Global attempt counter (max 4 Gemini calls)
- `unit/test_marketing_intelligence.py`: dependency-free tests for versioned playbook
  retrieval, research planning, source URL/metadata handling, source-diversity cautions,
  evidence/causal-inference boundaries, and the synthetic evaluation corpus structure.
- `evaluations/marketing_benchmark.json`: 33 synthetic research, marketing, provider,
  citation, and authorization tasks with expected properties and failure conditions.
  These definitions are not scored automatically; model-quality evaluation requires
  reviewed outputs, and live provider/authentication cases require isolated services.

The marketing-intelligence unit tests can run without the backend dependencies or database:

```bash
python -m unittest tests.unit.test_marketing_intelligence -v
```

## Test Data

Tests use mocked Gemini clients and database connections. No live API calls are made during test execution.
