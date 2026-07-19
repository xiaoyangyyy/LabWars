# Agent Organization Simulator Roadmap

> LabWars is the first benchmark environment in a broader Agent Organization MRI program. The academic lab is a pressure chamber, not the final domain boundary.

## 1. Generalization target

The reusable object is not the research lab. The reusable object is organizational dynamics:

```text
power + trust + memory + contribution + reputation + uncertainty + resource dependency
```

The same social-state model can be instantiated in many environments:

- academic laboratories
- companies and startups
- open-source communities
- political organizations
- game guilds
- online communities
- grant-review or peer-review networks

## 2. Why the lab remains the first environment

The academic lab is unusually dense as a benchmark because it contains:

- delayed rewards;
- ambiguous contribution measurement;
- reputation competition;
- power asymmetry;
- long-term collaboration;
- authorship and credit conflict;
- external review and funding pressure.

That makes it ideal for validating Agent MRI before moving to other organizations.

## 3. Environment abstraction

A future environment should specify:

| Layer | LabWars example | Generalized organization example |
|---|---|---|
| Agents | PI, PhD, postdoc, reviewer | manager, worker, maintainer, moderator |
| Resources | code, data, writing, PI access | budget, access, deployment rights, social capital |
| Rewards | authorship, publication, grant | promotion, ownership, reputation, governance power |
| Authority | PI, editor, funder | manager, board, maintainer, party leader |
| Events | authorship draft, rival preprint | roadmap dispute, policy vote, release deadline |
| Metrics | authorship dispute, trust fragmentation | conflict index, coalition stability, compliance gap |

## 4. Multi-layer organization path

Do not scale directly to 1000 weak agents. Scale first by hierarchy:

```text
Organization
|-- Department / Division
|   |-- Team / Lab
|   |   |-- Lead
|   |   |-- Senior member
|   |   |-- Junior member
|   |   `-- Contributor
|   |-- External evaluator
|   |-- Funding/resource authority
|   `-- Rival organization
```

This makes micro-to-macro emergence testable without sacrificing interpretability.

## 5. Benchmark family

A future benchmark suite can include:

1. LabWars: academic organization conflict
2. StartupWars: equity, product direction, and founder power
3. OpenSourceWars: maintainership, credit, governance, forks
4. GuildWars: status, raids, loot, leadership legitimacy
5. PolicyWars: coalition formation, rumor, authority, and public-private splits

All environments should share the same Agent Social State and ablation protocol.

## 6. Next engineering milestone

The next practical step is to keep LabWars as the reference environment while making the environment schema more portable:

- separate organization-level config from lab-specific config;
- keep SocialPotentialField domain-general;
- turn authorship-specific metrics into one family of credit-conflict metrics;
- add organization-ablation reports as a standard benchmark artifact.

## 7. Benchmark layer

The domain-general simulator roadmap is operationalized by the benchmark task protocol in [`docs/13-Agent-Social-Dynamics-Benchmark.md`](13-Agent-Social-Dynamics-Benchmark.md).
