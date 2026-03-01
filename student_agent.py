import re

# -----------------------------
# Domain detection
# -----------------------------
def detect_domain(scenario_context: str) -> str:
    s = scenario_context.lower()
    if "set of blocks" in s or "pick up a block" in s:
        return "blocks"
    if "set of objects" in s and "craves" in s:
        return "craves"
    return "generic"

# -----------------------------
# Prompt building
# -----------------------------
def keep_only_last_statement(scenario_context: str) -> str:
    parts = scenario_context.split("[STATEMENT]")
    header = parts[0]
    last_stmt = "[STATEMENT]" + parts[-1]
    return header + last_stmt

def inject_thought_marker(prompt_last_stmt: str) -> str:
    return re.sub(
        r"My plan is as follows:\s*\n\s*\[PLAN\]\s*$",
        "Let's think step by step briefly to find the optimal plan.\n\n[THOUGHT]",
        prompt_last_stmt,
        flags=re.M
    )

def extract_craves_facts(last_stmt: str):
    init_m = re.search(
        r"As initial conditions I have that,(.*?)(?:\nMy goal is to have that)",
        last_stmt,
        flags=re.S
    )
    init = (init_m.group(1).lower() if init_m else "")

    goal_m = re.search(
        r"My goal is to have that(.*?)(?:\n|$)",
        last_stmt,
        flags=re.S
    )
    goal = (goal_m.group(1).lower() if goal_m else "")

    harmony = "harmony" in init
    planet = sorted(set(re.findall(r"planet object ([a-z])", init)))
    province = sorted(set(re.findall(r"province object ([a-z])", init)))
    craves = sorted(set(re.findall(r"object ([a-z]) craves object ([a-z])", init)))
    goal_craves = sorted(set(re.findall(r"object ([a-z]) craves object ([a-z])", goal)))

    attackable0 = sorted(set(planet).intersection(province)) if harmony else []
    feastable0 = sorted([(x, y) for (x, y) in craves if harmony and (x in province)])

    return harmony, planet, province, craves, goal_craves, attackable0, feastable0

def build_prompt(scenario_context: str) -> tuple[str, str]:
    domain = detect_domain(scenario_context)
    prompt = scenario_context

    if domain == "craves":
        parts = prompt.split("[STATEMENT]")

        if len(parts) == 3:
            ex_stmt = "[STATEMENT]" + parts[1].split("My plan is as follows:")[0]
            h_ex, pl_ex, pr_ex, cr_ex, gc_ex, att_ex, fea_ex = extract_craves_facts(ex_stmt)

            ex_hints_block = (
                "\n[PARSED INITIAL FACTS]\n"
                f"harmony: {h_ex}\n"
                f"planet: {pl_ex}\n"
                f"province: {pr_ex}\n"
                f"craves: {cr_ex}\n"
                "[END PARSED INITIAL FACTS]\n"
                "[PARSED GOAL]\n"
                f"goal_craves: {gc_ex}\n"
                "[END PARSED GOAL]\n"
                "[STEP0 HINTS]\n"
                f"attackable0: {att_ex}\n"
                f"feastable0: {fea_ex}\n"
                "[END STEP0 HINTS]\n\n"
                "Let's think step by step briefly to find the optimal plan.\n\n"
                "[THOUGHT]\n"
                "I must analyze the hints. To achieve 'overcome', I need pain and province. If I lack an attackable object, I must use 'feast' to gain pain and shift province, then 'succumb' to restore harmony, repeating this until preconditions are met.\n"
                "[THOUGHT END]\n"
                "[PLAN]"
            )
            parts[1] = parts[1].replace("My plan is as follows:\n\n[PLAN]", ex_hints_block)

            tgt_stmt = "[STATEMENT]" + parts[2].split("My plan is as follows:")[0]
            h_tg, pl_tg, pr_tg, cr_tg, gc_tg, att_tg, fea_tg = extract_craves_facts(tgt_stmt)

            tgt_hints_block = (
                "\n[PARSED INITIAL FACTS]\n"
                f"harmony: {h_tg}\n"
                f"planet: {pl_tg}\n"
                f"province: {pr_tg}\n"
                f"craves: {cr_tg}\n"
                "[END PARSED INITIAL FACTS]\n"
                "[PARSED GOAL]\n"
                f"goal_craves: {gc_tg}\n"
                "[END PARSED GOAL]\n"
                "[STEP0 HINTS]\n"
                f"attackable0: {att_tg}\n"
                f"feastable0: {fea_tg}\n"
                "Hint: If attackable0 is empty, start with (feast ...) on available craves to shift province and gain pain, then (succumb ...) to regain harmony.\n"
                "[END STEP0 HINTS]\n\n"
                "Let's think step by step briefly to find the optimal plan.\n\n"
                "[THOUGHT]"
            )

            parts[2] = re.sub(
                r"My plan is as follows:\s*\n\s*\[PLAN\]\s*$",
                tgt_hints_block,
                parts[2],
                flags=re.M
            )

            prompt = parts[0] + "[STATEMENT]" + parts[1] + "[STATEMENT]" + parts[2]

    elif domain == "blocks":
        # Keep blocks logic working as it did
        prompt = keep_only_last_statement(scenario_context)
        prompt = inject_thought_marker(prompt)

    return prompt, domain

# -----------------------------
# Domain-specific system prompts
# -----------------------------
def get_system_prompt(domain: str) -> str:
    if domain == "craves":
        return (
            "You are an expert logical planning AI. Solve ONLY the final [STATEMENT] in the user message.\n"
            "The user prompt ends with '[THOUGHT]'. In max 3 sentences, map the goals to the necessary actions.\n"
            "Then write exactly '[THOUGHT END]' on its own line, then '[PLAN]'.\n\n"
            "CRITICAL HEURISTICS FOR THIS DOMAIN:\n"
            "1. THE 'OVERCOME' TARGET: To achieve a goal of 'X craves Y', the final step for that target is ALWAYS '(overcome X Y)'. This requires X to have 'Pain' and Y to have 'Province'.\n"
            "2. THE PAIN SOURCE: To give X 'Pain', you must perform '(attack X)' or '(feast X Z)'. The [STEP0 HINTS] list exactly which objects are 'attackable0'. Objects NOT in that list MUST use 'feast' instead.\n"
            "3. THE SUCCUMB TRAP (CRITICAL): NEVER perform '(succumb X)' immediately before '(overcome X Y)'. Succumb destroys the 'Pain' you need for Overcome! \n"
            "4. WHEN TO SUCCUMB: Only use '(succumb X)' AFTER an overcome, or if you desperately need to restore 'Harmony' to perform a completely DIFFERENT '(attack)' or '(feast)' on a new object.\n"
            "Write exactly one action per line in natural language. End with '[PLAN END]' on its own line."
        )

    if domain == "blocks":
        return (
            "You are an expert blocks-world planning AI. Solve ONLY the final [STATEMENT] in the user message.\n"
            "Write VERY concise reasoning (max 1 sentence) in [THOUGHT].\n"
            "Then output exactly '[THOUGHT END]' then '[PLAN]'.\n\n"
            "Output ONLY LISP actions, one per line, using ONLY these schemas:\n"
            "(engage_payload X)   ; pick up X\n"
            "(release_payload X)  ; put down X\n"
            "(unmount_node X Y)   ; unmount X from on top of Y\n"
            "(mount_node X Y)     ; stack/mount X on top of Y\n"
            "X,Y are lowercase block names like red, blue, orange, yellow, etc.\n"
            "Prefer the shortest valid plan; avoid unnecessary release_payload.\n"
            "Hint: If a block is on top of another block, use (unmount_node X Y) (NOT engage_payload X) to take it.\n"
            "End with '[PLAN END]'."
        )

    return (
        "You are an expert planning AI. Solve ONLY the final [STATEMENT].\n"
        "Write [THOUGHT] (max 1 sentence), then [THOUGHT END], then [PLAN].\n"
        "Output one action per line and end with [PLAN END]."
    )

# -----------------------------
# Output parser (domain-aware)
# -----------------------------
def parse_to_lisp(plan_text: str, domain: str):
    actions = []
    if "[PLAN]" in plan_text:
        plan_text = plan_text.split("[PLAN]")[-1]
    if "[PLAN END]" in plan_text:
        plan_text = plan_text.split("[PLAN END]")[0]

    for line in plan_text.strip().split("\n"):
        line = line.strip().lower()
        if not line: continue

        line = re.sub(r"^\s*(?:\d+[\).:-]\s*|[-*]\s*)", "", line).strip()
        if not line: continue

        if line.startswith("(") and line.endswith(")"):
            line = line[1:-1].strip()

        line = re.sub(r"[^\w\s]", " ", line)
        line = re.sub(r"\s+", " ", line).strip()
        if not line: continue

        line = line.replace("pick up", "engage_payload")
        line = line.replace("put down", "release_payload")
        line = line.replace("stack", "mount_node")

        words_to_remove = {"object", "the", "block", "from", "on", "top", "of", "onto", "another",
                           "planet", "province", "pain", "harmony", "table"}
        tokens = line.split()
        filtered = [t for t in tokens if t not in words_to_remove]
        if not filtered: continue

        verb = filtered[0]
        if domain == "craves":
            valid_verbs = {"attack", "succumb", "feast", "overcome"}
        elif domain == "blocks":
            valid_verbs = {"engage_payload", "release_payload", "unmount_node", "mount_node"}
        else:
            valid_verbs = {"engage_payload", "release_payload", "unmount_node", "mount_node",
                           "attack", "succumb", "feast", "overcome"}

        if verb not in valid_verbs: continue

        arity = {
            "attack": 1, "succumb": 1, "feast": 2, "overcome": 2,
            "engage_payload": 1, "release_payload": 1, "unmount_node": 2, "mount_node": 2,
        }.get(verb)

        if arity is not None and len(filtered) != 1 + arity: continue

        args = filtered[1:]
        if domain == "craves":
            if not all(re.fullmatch(r"[a-z]", a) for a in args): continue
        elif domain == "blocks":
            if not all(re.fullmatch(r"[a-z]+", a) for a in args): continue

        actions.append(f"({' '.join(filtered)})")

    return actions

def canonicalize_blocks(actions: list[str]) -> list[str]:
    out = []
    for a in actions:
        if out and a.startswith("(unmount_node "):
            m = re.match(r"^\(unmount_node\s+([a-z]+)\s+([a-z]+)\)$", a)
            if m:
                x = m.group(1)
                prev = out[-1]
                if prev == f"(engage_payload {x})":
                    out.pop()
        out.append(a)
    return out

def parse_blocks_init(last_stmt: str):
    init_m = re.search(r"As initial conditions I have that,(.*?)(?:\nMy goal is to have that)", last_stmt, flags=re.S)
    init = (init_m.group(1).lower() if init_m else "")
    on_pairs = re.findall(r"the\s+([a-z]+)\s+block\s+is\s+on\s+top\s+of\s+the\s+([a-z]+)\s+block", init)
    on_table = set(re.findall(r"the\s+([a-z]+)\s+block\s+is\s+on\s+the\s+table", init))
    return on_pairs, on_table

def parse_blocks_goal(last_stmt: str):
    goal_m = re.search(r"My goal is to have that(.*?)(?:\n\nMy plan is as follows:)", last_stmt, flags=re.S)
    goal = (goal_m.group(1).lower() if goal_m else "")
    goal_pairs = set(re.findall(r"the\s+([a-z]+)\s+block\s+is\s+on\s+top\s+of\s+the\s+([a-z]+)\s+block", goal))
    return goal_pairs

def extract_block_names(last_stmt: str) -> set[str]:
    return set(re.findall(r"the\s+([a-z]+)\s+block", last_stmt.lower()))

def canonicalize_blocks_stateful(actions: list[str], last_stmt: str, max_insertions: int = 6) -> list[str]:
    on_pairs, on_table = parse_blocks_init(last_stmt)
    goal_pairs = parse_blocks_goal(last_stmt)
    block_names = extract_block_names(last_stmt)

    on = {}
    for x, y in on_pairs: on[x] = y
    for x in on_table: on[x] = "table"

    holding = None
    out = []
    inserted = 0

    def insert(action: str):
        nonlocal inserted
        out.append(action)
        inserted += 1

    def ensure_empty_hand():
        nonlocal holding
        if holding is not None:
            insert(f"(release_payload {holding})")
            on[holding] = "table"
            holding = None

    def take_block(x: str):
        nonlocal holding
        if holding == x: return
        ensure_empty_hand()

        below = on.get(x, "table")
        if below is None:
            holding = x
            return

        if below != "table":
            insert(f"(unmount_node {x} {below})")
            holding = x
            on[x] = None
        else:
            insert(f"(engage_payload {x})")
            holding = x
            on[x] = None

    def goal_align_mount(x: str, y: str):
        if (x, y) not in goal_pairs and (y, x) in goal_pairs:
            return y, x
        return x, y

    original = actions[:]

    for a in actions:
        if " table" in a: continue

        if out and a.startswith("(unmount_node "):
            m = re.match(r"^\(unmount_node\s+([a-z]+)\s+([a-z]+)\)$", a)
            if m:
                x = m.group(1)
                if out[-1] == f"(engage_payload {x})":
                    out.pop()

        m1 = re.match(r"^\((engage_payload|release_payload)\s+([a-z]+)\)$", a)
        m2 = re.match(r"^\((unmount_node|mount_node)\s+([a-z]+)\s+([a-z]+)\)$", a)

        if m1:
            verb, x = m1.group(1), m1.group(2)
            if x not in block_names: continue

            if verb == "engage_payload":
                below = on.get(x, "table")
                if below != "table" and below is not None:
                    take_block(x)
                else:
                    take_block(x)
            else:
                if holding == x:
                    out.append(a)
                    on[x] = "table"
                    holding = None
            continue

        if m2:
            verb, x, y = m2.group(1), m2.group(2), m2.group(3)
            if x not in block_names or y not in block_names: continue

            if verb == "unmount_node":
                actual_below = on.get(x, "table")
                if actual_below not in (None, "table") and actual_below != y:
                    y = actual_below
                take_block(x)
                continue

            if verb == "mount_node":
                x, y = goal_align_mount(x, y)
                take_block(x)
                out.append(f"(mount_node {x} {y})")
                on[x] = y
                holding = None
                continue

        continue

    if inserted > max_insertions:
        return original

    return out

# ==============================================================================
# STUDENT AGENT CLASS
# ==============================================================================
class AssemblyAgent:
    def __init__(self):
        # La clase ya no necesita guardar el system_prompt genérico
        # porque usamos get_system_prompt() dinámicamente.
        pass
        
    def solve(self, scenario_context: str, llm_engine_func) -> list:
        """
        Recibe el texto del escenario y la funcion del motor LLM.
        Debe retornar una lista de strings con las acciones extraidas en formato LISP.
        """
        # 1. Construir el prompt y detectar el dominio
        prompt, domain = build_prompt(scenario_context)
        system_prompt = get_system_prompt(domain)
        
        # 2. Llamada al LLM usando el motor oficial
        output_text = llm_engine_func(
            prompt=prompt,
            system=system_prompt,
            max_new_tokens=160,
            temperature=0.0,
            top_p=1.0,
            do_sample=False
        )
        
        # 3. Extraer solo el bloque PLAN
        plan_text = output_text
        if "[PLAN]" in output_text:
            plan_text = output_text.rsplit("[PLAN]", 1)[-1]
            if "[PLAN END]" in plan_text:
                plan_text = plan_text.split("[PLAN END]")[0]
                
        # 4. Parsear con las reglas específicas de dominio
        predicted = parse_to_lisp(plan_text, domain)
        
        # 5. Aplicar canonicalización (tu lógica experta de state machine) si es blocks
        if domain == "blocks":
            predicted = canonicalize_blocks(predicted)
            last_stmt = keep_only_last_statement(scenario_context)
            predicted = canonicalize_blocks_stateful(predicted, last_stmt)
            
        return predicted