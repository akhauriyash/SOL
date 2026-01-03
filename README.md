# Self Optimizing Language Models

![SOL Main Figure descriing how training and inference work.](SOLMainFig.png)

Work on efficient Large Language Model (LLM) inference has emphasized how to make **every decoding step cheaper** (quantization, sparsity), but less so on **how much compute each token should receive**. This leaves deployments with a rigid, per-token budget that over-computes on easy tokens and under-computes on hard ones. We study \emph{dynamic budget allocation} for language models: learning how much compute to use for every generated token. Concretely, we train a small policy network ($0.5\%$ of the LLM size) that reads the LLM’s hidden state and selects a discrete action $\kappa \in \{\kappa_1,\dots,\kappa_A\}$ which controls the amount of compute allocated at every decode step. The base LLM weights are unchanged. This turns inference efficiency optimization into a sequential decision problem. We show that a learned policy can allocate dense context when it matters and sparse context when it does not. Further, our method can teach a policy to jointly optimize for quantization, sparsity and pruning. Self-Optimizing Language Models (SOL) consistently out-perform static budget-allocation strategies. SOL achieves a win-rate of 97.8\% over fixed-budget allocation, unlocking a complementary, underexplored dimension for efficiency optimization.

## Policy Models
All policy models tested in our paper as well as full csv of results + plotting code can be found on Google Drive as [SOL Model + Results](https://drive.google.com/file/d/1471okg2V8352uOwbIeqxtKi1nFgCRP4H/view?usp=sharing)


## Configuration
All run-time hyperparameters are defined in `utils/config.py` via the `Config` dataclass. You can override any field by editing the config files in `official_configs/`

Key config controllers:

```yaml
model_name: meta-llama/Llama-3.2-1B
algo: grpo                      # grpo | sft

## Efficiency / action space
sparsity_criteria: quest         # recency | relevancy | quest
quest_page_size: 8               # page size for QUEST (only used if sparsity_criteria=quest)
relevancy_tier: per_head         # per_head | per_layer (only used if sparsity_criteria=relevancy)

keep_fracs: [0.2, 1.0]           # token-attention keep fractions κ (1.0 = dense)
struct_prune_choices:            # structured MLP prune actions: "s{keep%}" (e.g., s60 => keep 60%)
  - s100
  - s80
  - s60
quant_choices:                   # activation quant actions: "q{bits}" (e.g., q5..q16; q16 = dense)
  - q8
  - q16
enable_prune_quant: true         # if false, prune/quant are held at dense settings

## Budget conditioning
# Budgets are sampled per input sequence. For each axis, set either a discrete list
# (budget_*_list) or a uniform range (budget_*_min/max). If unset, defaults below are used.

budget_tok_list: null
budget_tok_min: 0.1
budget_tok_max: 1.0

budget_prune_list: null
budget_prune_min: 0.4
budget_prune_max: 1.0

budget_q_ratio_list: null
budget_q_ratio_min: 0.3125       # 5/16
budget_q_ratio_max: 1.0          # 16/16

C_target: 0.5                    # default token keep target (alias: keep_target / C_target_token) used for eval
C_target_prune: 0.70             # default prune keep target ρ used for eval
C_target_quant_bits: 9.0         # default quant target in bits (internally uses q_ratio = bits/16) used for eval

## Reward / GRPO
task_w_kl: 0.0                   # 0.0 = log-likelihood; 1.0 = KL-to-dense baseline
reward_agg: sum                  # null | sum | max
reward_gamma: 0.85
grpo_level: process              # process | outcome
grpo_rollouts_per_input: 16      # K
grpo_norm: center                # center | zscore
adv_whiten_global: true

## Episode + always-dense tokens
rollout_len: 16                  # control horizon T
context_len: 1024                # dense prefill length
Ts: 4                            # sink tokens (always dense)
Tw: 2                            # window tokens (always dense)
pi_temperature: 1.3              # rollout sampling temperature
```

## Training

Example scripts are provided in `scripts.sh`. Single training example
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29513 train.py   \
  --wandb_project SOL_RLS_MSC     --wandb_run_name Llama8Bi                       \
   --config < path to config >/official_configs/All_Variants_Llama8Bi.yml
```

## Evaluation

To evaluate on perplexity metrics, refer to `eval_script.sh`. It automatically executes random and fixed baselines.

To evaluate on downstream tasks, run:

```
python eval_policy_lmeval.py   --ckpt_dir <base_path>/SOL/checkpoints/Llama8Bi-20260102-163733  \
  --mode latest   --tasks hellaswag,squadv2,arc_easy,winogrande   --batch_size 8  \
  --episode_len 16 --sparsity_bias 0.0 --quant_bias 0.0 --prune_bias 0.0          \
  --greedy_policy --fixed_from_policy --export_sparsity_json base_path.json
```

This will export all results to the json path in `export_sparsity_json`.

All results used in the paper are provided in `results/` of this repository.

Checkpoints are saved as `'policy_{text}.pt'`, where text can be checkpoint step, latest, etc. Switch between checkpoints by changing 'mode' argument.

To compare generations from policy vs dense model, run the command below, with a relatively long string in input_sentence (so that token-sparsity has some token-based pages to work with).

```
python gen_policy_vs_dense.py \
  --ckpt_dir <base_path>/SOL/checkpoints/RL_LCE_Quest4-20251028-213126 \
 --input_sentence "Self optimizing language models represent one of the most intriguing directions in the evolution of artificial intelligence. These systems are not only capable of generating text or analyzing data, but they can also refine their own internal para meters and strategies over time. The idea is that instead of relying solely on human engineers to adjust their learning methods, the models themselves learn how to learn better. This involves a loop of int rospection and adaptation in which the model examines its own performance, identifies inefficiencies, and modifies the way it processes information. In this sense, a self optimizing model behaves less like a static piece of software and more like a living system that evolves based on its experiences. One of the core ideas behind this kind of optimization is meta learning. Meta learning refers to learning ho w to improve learning itself. A model that can perform meta learning can look at patterns in its previous training sessions and use that insight to predict which learning strategies will yield better resul ts in the future. For example, if the model detects that certain configurations of attention weights consistently lead to faster convergence during fine tuning, it can prioritize those configurations next time. This recursive loop of optimization can lead to rapid improvements, especially when combined with large scale data and distributed computing. A key challenge in self optimization lies in ensuring sta bility. A model that continuously rewrites its own training strategy could easily spiral into inefficient or even destructive feedback loops if not carefully constrained. To prevent this, researchers often introduce a form of regulatory oversight within the system. The model might be allowed to propose updates to its learning algorithm, but those proposals are tested in controlled environments before being applied at scale. In this way, self optimizing models combine autonomy with safety, balancing exploration with caution. The ultimate goal is to create systems that are both flexible and reliable, capable o f improving themselves without losing alignment with human goals. Another interesting aspect of this concept is resource efficiency. A self optimizing language model can learn to manage its computational r esources more intelligently. It might recognize that certain patterns in the data are redundant and therefore allocate less processing power to them. It could also identify opportunities to compress its in ternal representations without significant loss of accuracy. By doing so, the model becomes not only smarter but also more efficient, reducing the cost of running large scale inference and training operati ons. This efficiency has direct implications for accessibility, as it could make powerful models available to smaller organizations and researchers who cannot afford massive infrastructure. In the context of language understanding, self optimization can also lead to more coherent and context aware outputs. As the model refines its internal representations, it develops a deeper sense of linguistic structure, semantics, and pragmatics. It can better anticipate the needs of the user, producing responses that are more relevant and precise. Over time, such models could reach a level of adaptability that allows th em to tune their style, tone, and reasoning depth based on subtle cues in conversation. This represents a shift from fixed style generation to dynamic communication that evolves in real time. Some research ers propose that future versions of these systems might integrate continuous feedback loops from the real world. In such setups, the model would receive ongoing signals from users, sensors, or external eva luators. These signals would serve as reinforcement for desired behaviors and corrections for undesirable ones. The model would then incorporate this feedback to refine its objectives and internal dynamics . Essentially, the model becomes its own experimenter, using every interaction as a data point for improvement. This form of autonomous optimization could lead to breakthroughs in areas like adaptive tutor ing, scientific discovery, and creative collaboration. There is also a philosophical angle to self optimizing language models. As these systems become increasingly capable of revising their own reasoning s tructures, the distinction between programmed intelligence and emergent intelligence begins to blur. A self optimizing model is not just executing instructions but rewriting its own playbook. This raises q uestions about agency, control, and accountability. If a system modifies its own reasoning mechanisms in ways not explicitly anticipated by its creators, who is responsible for the consequences? Addressing these questions will require careful ethical and regulatory frameworks, as well as transparency in how self optimization is implemented and monitored. Despite these challenges, the potential benefits are immense. A model that can adapt faster than it is retrained could dramatically accelerate innovation. It could tailor itself to specific domains in hours rather than weeks, or learn new languages without e xplicit retraining. The concept aligns with the broader vision of lifelong learning in artificial systems, where models continuously evolve in response to changing environments and goals. In practical term s, this could lead to language models that become better collaborators, advisors, and problem solvers over time. The long term trajectory of this research points toward models that are not just tools but p artners in discovery. A self optimizing language model could help design its own future versions, proposing architectural " \
  --generation_tokens 256   --temperature 0.7 --top_p 0.9
```

Sample completion:

```
--- True Dense continuation ---

izations and exploring new possibilities for language understanding. This could lead to a more dynamic, interactive, and nuanced understanding of language, one that evolves in real time and adapts to the needs of its users. In conclusion, self optimizing language models represent a fascinating and ambitious area of research. They offer the potential to revolutionize language understanding, language learning, and human-machine collaboration. The concept of meta learning, which involves learning how to learn, is central to this field. Meta learning allows models to adjust their own learning strategies, leading to rapid improvements in performance. The concept of regulatory oversight, which ensures that models remain stable and safe, is also critical. Finally, the concept of resource efficiency highlights the importance of optimizing computational resources while maintaining accuracy. The potential benefits of self optimizing language models are immense, and the challenges they present are equally exciting. As this research continues to evolve, we can expect to see more breakthroughs in language understanding, language learning, and human-machine collaboration. The future of self optimizing language models holds tremendous promise for advancing the state of the art in artificial intelligence. The possibilities are endless, and the journey is just beginning.
In this section, we will explore the fascinating world of self optimizing language models. We will delve into the concept of meta learning, regulatory


[chars: 1527]

--- Policy (sparse) continuation ---

izations or adaptations that would enhance its capabilities or decrease its cost. It could explore new ways of understanding language, learning new ways of learning, and make choices that improve understanding and learning at scale. In summary, self optimizing language models represent a fascinating frontier in artificial intelligence. They are capable of autonomous learning, dynamic adaptation, and meta learning. The possibilities are endless, and the journey ahead is full of promise.defining a new frontier for artificial intelligence, in a world where it is increasingly important to understand the impact of these systems on human society and the environment. With each step forward, we can only expect to see the boundaries of language modeling technology become increasingly blurred, and the potential for breakthroughs in areas like creativity, collaboration, and innovation become increasingly possible. A self optimizing language model is a complex system that requires careful design and implementation. Researchers are exploring various techniques to make this a reality. One key aspect of this research is the design of the training process. A self optimizing language model needs to be able to adapt to changing environments and goals. To do this, the model must be able to assess its own performance and make adjustments accordingly. This involves a delicate balance between exploration and exploitation. To achieve this, researchers are experimenting with various techniques to test different types


[chars: 1518]

============================

Achieved Token-Keep-Rate        67.18750012223609%
Achieved Prune-Keep             74.92187649477273%
Achieved Quant-Ratio            0.53125 (1 = full 16-bit)

============================
```