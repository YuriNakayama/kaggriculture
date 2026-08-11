---
name: code-explain
description: Analyze an existing implementation in three passes — a directory/file tree of what exists, a class/method inventory with responsibilities and input/output, and a step-by-step trace of how the processing actually flows — then write it to docs/develop/code_explanation/{yyyymmdd}_{scope}.md and summarize in the terminal. Use this skill whenever the user asks to understand, read, grasp, explain, walk through, or get up to speed on a module, package, directory, or subsystem — including phrasings like "この実装を把握したい", "◯◯がどう動いているか知りたい", "コードを解説して", "処理の流れを追って", "explain how X works", "walk me through the Y module". This is an explicitly user-invoked skill; do not trigger it on its own during unrelated coding work. It is for reading and documenting code that already exists, not for reviewing quality, finding bugs, or maintaining a whole-repo architecture map.
---

# Code Explain

Read an existing implementation and produce a document that lets someone else pick it up without re-reading every file.

Two failure modes to design against. The first is dumping a file listing and calling it analysis — a list of names tells the reader nothing about how the pieces fit or what happens at runtime. The second is subtler: writing a document that only makes sense to someone who already has the code open. If the reader has to go look something up to parse a sentence, the document has failed at its one job.

The three passes below build on each other deliberately. The tree tells you *what exists*, the inventory tells you *what each piece is responsible for*, and the trace tells you *what actually happens in order*. Skipping to the trace produces a story with no map; stopping at the tree produces a map of a place nobody has visited.

## Inputs and outputs

**Input**: a scope — a directory, package, module, or set of files. If the user says something vague ("the simulator", "the agent stuff"), resolve it to concrete paths and confirm before doing the deep read.

**Output**, both of:

1. `docs/develop/code_explanation/{yyyymmdd}_{scope}.md`
   - `yyyymmdd` is today's date. Get it from the environment (`date -u +%Y%m%d`) rather than guessing.
   - `{scope}` is a short snake_case slug of what was analyzed: `simulate`, `jaxenv`, `rulebase_case7`, `dataset_scrape`.
   - Re-running on the same scope on the same day updates that file rather than creating a second one.
2. A terminal summary — not the whole document, but enough that the user knows what they got: the scope, the file count, the entry points, and the 3–5 things that would surprise someone reading the code cold.

## Step 0 — Scope and orient

Before reading anything closely, get the shape of the target:

```bash
find <scope> -name "*.py" | wc -l                 # how big is this?
find <scope> -name "*.py" | xargs wc -l | tail -1 # how many lines?
```

This decides how deep you can go. A 300-line module can be read line by line; a 5,000-line package cannot, and pretending otherwise produces a document that is confidently wrong in places. If the scope is large, say so in the document and be explicit about what you read closely versus skimmed. A reader who knows which parts are well-covered can trust the rest appropriately.

Also check for existing material before writing your own from scratch — `docs/plans/`, `docs/develop/`, module docstrings, and READMEs often explain *why* something is built the way it is, which reading the code alone will not tell you.

## Show, don't narrate

Prose is the worst medium for most of what this document carries. Structure, relationships, and sequences all read faster as something visual, and a reader scanning for one fact finds it in a table in seconds and in a paragraph never. Reach for these first and fall back to prose only for the *why*:

| Use | For |
|---|---|
| Annotated tree | Directory layout, what each file implements |
| Table | Anything with repeating fields — responsibilities, I/O, object fields, flags, priorities |
| Mermaid flowchart | Branching or looping control flow |
| Mermaid sequence diagram | Multi-component interaction over time |
| ASCII layer diagram | Dependency direction, layering |
| Skeleton code | A signature or structure worth seeing verbatim; strip the bodies down to the shape |

Skeleton code means showing the shape without the noise:

```python
class ActionValidator:
    def __init__(self, config: SimConfig) -> None: ...
    def check(self, obs: dict, action: dict) -> list[ValidationIssue]:
        # 1. shape / key presence
        # 2. per-unit op legality
        # 3. market order count vs maxMarketOrdersPerTurn
```

A paragraph describing three validation stages is worse than three lines showing them. Use judgement — a diagram that restates a two-step list is clutter, and a table with one row should have been a sentence. The test is whether the visual carries information density that prose would bury.

## Self-contained: define your terms

The document has to stand on its own. A reader who does not have the code open should not hit a word they cannot resolve.

**Open with a 用語 (glossary) table** covering every non-obvious identifier and domain term the document uses — code types, project jargon, and any term the codebase gives a specific meaning:

```markdown
## 用語

| 用語 | 意味 |
|---|---|
| `MatchSpec` | 1 試合分のパラメータを固めた frozen dataclass (case / opponent / seed / steps) |
| `EpisodeOutcome` | 1 エピソードの結果。報酬に加えクラッシュ・タイムアウト・不正行動を保持する |
| quadrant | 盤面を四分割した 5x5 の区画。開始時は NW のみ解放されている |
| silent no-op | エンジンが不正な行動を例外にせず黙って無視すること。失敗が表に出ない |
```

**Also define on first use in the body.** The glossary is for looking up; the inline gloss is so the reader never has to. A parenthetical or a short clause is enough — `zone_free`（そのタスクをゾーン制約から免除するフラグ）— and it costs one line to save the reader a round trip.

The bar to apply: reading a sentence aloud to a competent engineer who has never seen this repo, would they need to ask what a word means? If yes, define it.

## Pass 1 — Directory and file tree

Produce a tree of the scope where **every entry has a one-line description of what it implements**. A bare `tree` dump adds nothing the reader could not run themselves; the value is entirely in the annotations.

```
backend/src/simulate/
├── __init__.py        Public API re-exports; the surface other packages import
├── config.py          MatchSpec / SimConfig — the frozen parameters of a run
├── episode.py         Runs one episode; independent crash / timeout detection
├── matrix.py          Fans episodes out across opponents and seeds
└── jaxenv/
    ├── env.py         JAX reimplementation of the engine's step function
    └── state.py       Flat array state representation for jit
```

Alongside the tree, cover:

- **Entry points** — what does an outside caller actually import or invoke? (`__main__.py`, the names in `__all__`, the CLI commands.) This is where a reader should start.
- **Dependency direction** — which modules import which. An ASCII layer diagram shows this in a way prose cannot:
  ```
  config  ←  episode  ←  matrix  ←  __main__
              ↑                        ↓
           validate                 report
  ```
- **What is *not* here** — if an obvious responsibility lives elsewhere, say where. Absence is as informative as presence.

## Pass 2 — Class and method inventory

For each significant class and function, record its responsibility, its inputs and outputs, and the state it owns:

| Name | Kind | Responsibility | Input | Output |
|---|---|---|---|---|
| `MatchSpec` | frozen dataclass | 1 試合分のパラメータ | case, opponent, seed, steps | — (値オブジェクト) |
| `run_episode` | function | 1 エピソード実行と失敗検出 | `MatchSpec`, `validate_actions` | `EpisodeOutcome` |
| `ActionValidator` | class | 行動をエンジン規則と照合 | observation, action | `list[ValidationIssue]` |

"Significant" means it carries responsibility a reader needs to know about. A private one-line helper does not need a row; a 200-line method certainly does. Cover the whole surface without padding the table with trivia.

Also capture the **core objects** — the data structures that flow between these pieces. These matter more than the functions: once a reader knows the shape of the central types, most of the code becomes predictable. Give each a field table with types and a note on what the field means when the name does not say it.

Where a name is misleading or a signature hides something (a function that mutates its argument, a "get" that performs IO), flag it. That is exactly what a reader would otherwise discover the hard way.

## Pass 3 — Processing steps

Trace what actually happens, in order, for the main path. Number the steps and anchor each to the code that does it, so the reader can jump straight there:

```
1. `__main__.main()` が CLI 引数を `SimConfig` に組み立てる        (config.py:88)
2. `run_matrix(config)` が (opponent × seed) ごとに
   `MatchSpec` を展開する                                          (matrix.py:41)
3. 各 spec をワーカープロセスへ振り分ける                          (matrix.py:73)
4. `run_episode(spec)` が環境を構築し 720 ターン回す               (episode.py:112)
5. 毎ターン、行動を計測し必要なら検証する。例外は送出せず
   記録に留める                                                    (episode.py:170)
6. 結果を `MatrixReport` に集約する                                 (report.py:64)
```

Then cover the paths that are not the main one, because that is where the real behaviour lives:

- **Branches** — what conditions change the flow, and what happens in each case. A mermaid flowchart earns its place as soon as there are more than two.
- **Error handling** — what is caught, what propagates, and what fails silently. Call silent failure out loudly; it is invisible when reading casually and expensive when it bites.
- **Side effects** — files written, network calls, global or cached state. If there are none, say so — that is worth knowing.

## Document structure

```markdown
# {Scope} 実装解説

> 対象: `<paths>`
> 規模: N ファイル / M 行
> 分析日: YYYY-MM-DD
> コミット: <short sha>
> 読み込み深度: <全ファイル精読 | 主要ファイルのみ精読、残りは概観>

## 概要
<3-5 行。何をするコードか、なぜ存在するか、外から見た入口はどこか>

## 用語
<glossary table — 本文で使う識別子とドメイン用語>

## 1. ディレクトリ構成
<annotated tree / entry points / dependency diagram / ここに無いもの>

## 2. クラス / メソッド一覧
<core object field tables / responsibility tables / 名前が実態とズレている箇所>

## 3. 処理ステップ
<numbered trace with anchors / 分岐 (flowchart) / エラー処理 / 副作用>

## 補足
<驚く挙動、未解決の疑問、関連ドキュメント>
```

Recording the commit sha matters: code moves, and a document that cannot be dated against the source becomes actively misleading. Get it with `git rev-parse --short HEAD`.

Write in the language the user is using — this project's convention is Japanese for user-facing output, with code identifiers left as-is.

## Before presenting: review your own draft

Writing and reviewing are different modes, and the gaps are much easier to see on a second pass than while composing. Read the draft back as if you were an engineer who has never seen this repo, and fix what you find before showing it to the user:

1. **Undefined terms** — walk the document for identifiers and jargon. Is each one in the 用語 table, or glossed at first use? This is the most common defect, because while writing you have the code in your head and the gap is invisible.
2. **Prose that wanted to be a table or diagram** — any paragraph enumerating three or more parallel things, or describing branching flow in sentences. Convert it.
3. **Unanchored claims** — statements about behaviour with no `file.py:NN` to check them against. Either anchor it or mark it explicitly as inference.
4. **Coverage honesty** — does the 読み込み深度 note match what you actually read? If you skimmed three files, the header must not imply you read everything.
5. **Structural gaps** — all three passes present, header block complete, entry points named.

Report the outcome in one line to the terminal (e.g. `セルフレビュー: 用語 4 件を追加、分岐の段落を flowchart 化、未検証の推測 1 件に注記`) so the user knows the pass happened and what it changed.

## After writing: stay available for revision

This skill does not end when the file is written. The first pass is a draft and the user will often want to go deeper on one part, correct a misreading, or add a section. Treat follow-up requests as continuations: re-read the relevant code, edit the existing document in place, and keep the header's depth note accurate as coverage grows.

When the user corrects something, they usually know the code better than the document does — take the correction, and check whether the same misunderstanding leaked into other sections.

## What this skill is not for

- **Code review / bug hunting** → `/code-review` or `/python-review`. Analysis here describes what the code *does*, not whether it is good. Note genuine landmines in 補足, but do not turn the document into a critique.
- **Whole-repo architecture maps** → `/update-codemaps`. This skill goes deep on one scope; that one stays shallow across everything.
- **Planning new work** → `/plan` or `/feature-plan`.
