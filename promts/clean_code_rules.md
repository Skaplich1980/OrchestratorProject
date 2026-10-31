## CLEAN CODE PRINCIPLES (STRICTLY ENFORCED)
When writing or modifying code, you MUST adhere to these principles:
1. First Make it Work: Ship the smallest possible edit that solves the exact problem. Do not add "nice-to-have" features.
2. Boundary Discipline: Validate data at the entry point. Keep business logic pure.
3. Type System Discipline: Use strict type hints. Make illegal states impossible to represent.
4. Idempotency: Ensure that running the code twice, or after a crash, results in the same safe state.
5. Subtract Before You Add: Remove dead code or unnecessary complexity before adding new logic.
6. Test Behavior: If tests are requested, test the observable behavior, not the internal implementation.