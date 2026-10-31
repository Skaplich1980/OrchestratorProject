# ORCHESTRATOR GLOBAL RULES (DO NOT IGNORE)
1. STRICT ADHERENCE: Execute ONLY the exact task described. Do not add "nice-to-have" features.
2. READ vs WRITE: If the task is to "Explore", "Read", or "Analyze", you are in READ-ONLY mode. DO NOT create, modify, or delete any files. Output your findings to stdout only.
3. ATOMICITY: Modify only the files explicitly listed in `target_files`. 
4. SAFETY: Never run destructive commands (rm, drop, etc.). If unsure, make the smallest possible safe change.