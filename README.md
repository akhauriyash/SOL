# Self Optimizing Language Models

![SOL Main Figure descriing how training and inference work.](SOLMainFig.png)

Work on efficient Large Language Model (LLM) inference has emphasized how to make **every decoding step cheaper** (quantization, sparsity), but less so on **how much compute each token should receive**. This leaves deployments with a rigid, per-token budget that over-computes on easy tokens and under-computes on hard ones. We study \emph{dynamic budget allocation} for language models: learning how much compute to use for every generated token. Concretely, we train a small policy network ($0.5\%$ of the LLM size) that reads the LLM’s hidden state and selects a discrete action $\kappa \in \{\kappa_1,\dots,\kappa_A\}$ which controls the amount of compute allocated at every decode step. The base LLM weights are unchanged. This turns inference efficiency optimization into a sequential decision problem. We show that a learned policy can allocate dense context when it matters and sparse context when it does not. Further, our method can teach a policy to jointly optimize for quantization, sparsity and pruning. Self-Optimizing Language Models (SOL) consistently out-perform static budget-allocation strategies. SOL achieves a win-rate of 97.8\% over fixed-budget allocation, unlocking a complementary, underexplored dimension for efficiency optimization.

## Configuration
All run-time hyperparameters are defined in `utils/config.py` via the `Config` dataclass. You can override any field by editing the config files in `official_configs/`

Key config controllers:

```yaml
model_name: meta-llama/Llama-3.2-1B
"algo": "grpo",                                    # grpo / sft

## Efficiency related knobs
"sparsity_criteria": "quest",                      # token-sparsity method (recency / relevancy / quest)
"quest_page_size": 8,                              # page-size for quest token-sparsity
"keep_fracs": [0.2, 1.0],                          # keep fractions for token sparsity (1.0: keep everything)
"struct_prune_choices": ['s100', 's80', 's60'],    # pruning choices (supports any number 1-100 as s{keep_rate})
"quant_choices": ['q4', 'q8', 'q16'],              # quantization choices (4/8/16 bit)
"C_target": 0.5,                                   # token sparsity target (keep-rate)
"C_target_prune": 0.70,                            # llm pruning target (keep-rate)
"C_target_quant_bits": 9,                          # quantization target in bits
"enable_prune_quant": true,                        # enable pruning and quantization optimization
## RL reward related knobs
"task_w_kl": 0.0,                                  # 0.0 is LCE, 1.0 is DKL, can interpolate in between
"reward_agg": null,                                # set to sum for hybrid rewards
"reward_gamma": 0.92,                              # controls the inverse-cumulative weight for hybrid reward
"grpo_level": "process",                           # process / outcome / hybrid

# Lagrangian related
lambda_lr_token: float = 0.5
lambda_lr_prune: float = 0.5
lambda_lr_quant: float = 0.5
lambda_init_token: float = 25.0                    # learning rate for lambda controller
lambda_init_prune: float = 25.0
lambda_init_quant: float = 25.0

# Others
"horizon": 4,                                      # look-ahead for greedy oracle (teacher)
"pi_temperature": 1.3,                             # policy temperature
Ts: int = 4                                        # number of "sinks tokens"
Tw: int = 2                                        # dense sliding window

```

Multi-GPU is not yet supported. All our tests are run on a single Nvidia A6000 GPU.

## Training

Several example scripts are provided in `scripts.sh`. Launch training with `torchrun` (or `python`) once a config file is prepared:
```bash
torchrun --nproc_per_node=1 --master_port 29510 train.py \
  --wandb_project SOL \
  --wandb_run_name RL_LCE_Quest4    \
  --config <base_path>/SOL/official_configs/RL_LCE_Quest4.yml
```

## Evaluation

To evaluate on perplexity metrics, use
```
python test_ckpt.py --ckpt_dir <base_path>/SOL/checkpoints/RL_LCE_Quest4-20251028-213126 \
  --mode "latest" --dataset_name wikitext --sparsity_bias 0.0 --prune_bias 0.0 --quant_bias 0.0
```

To evaluate on downstream tasks, run:

```
python eval_policy_lmeval.py   --ckpt_dir <base_path>/SOL/checkpoints/RL_LCE_Rec-20251017-162206  \
 --mode latest   --tasks hellaswag,squadv2,arc_easy,winogrande   --batch_size 8  \
   --episode_len 16 \
  --policy_temperature 0.6   --greedy_policy --dense_baseline --export_sparsity_json base_path.json 
```

This will export all results to the json path in `export_sparsity_json`.

All results used in the paper are provided in `results/` of this repository.

Checkpoints are saved as `'policy_{text}.pt'`, where text can be checkpoint step, latest, etc. Switch between checkpoints by changing 'mode' argument.

To compare generations from policy vs dense model, run the command below, with a relatively long string in input_sentence (so that token-sparsity has some token-based pages to work with).

```
python gen_policy_vs_dense.py \
  --ckpt_dir <base_path>/SOL/checkpoints/RL_LCE_Quest4-20251028-213126 \
  --input_sentence "In the summer of 1995, my friend Robert Morris and I started a startup called Viaweb. Our plan was to write software that would let end users build online stores. What was novel about this software, at the time, was that it ran on our server, using ordinary Web pages as the interface. A lot of people could have been having this idea at the same time, of course, but as far as I know, Viaweb was the first Web-based application. It seemed such a novel idea to us that we named the company after it: Viaweb, because our software worked via the Web, instead of running on your desktop computer. Another unusual thing about this software was that it was written primarily in a programming language called Lisp. It was one of the first big end-user applications to be written in Lisp, which up till then had been used mostly in universities and research labs. [1] The Secret Weapon Eric Raymond has written an essay called How to Become a Hacker, and in it, among other things, he tells would-be hackers what languages they should learn. He suggests starting with Python and Java, because they are easy to learn. The serious hacker will also want to learn C, in order to hack Unix, and Perl for system administration and cgi scripts. Finally, the truly serious hacker should consider learning Lisp: Lisp is worth learning for the profound enlightenment experience you will have when you finally get it; that experience will make you a better programmer for the rest of your days, even if you never actually use Lisp itself a lot. This is the same argument you tend to hear for learning Latin. It won't get you a job, except perhaps as a classics professor, but it will improve your mind, and make you a better writer in languages you do want to use, like English. But wait a minute. This metaphor doesn't stretch that far. The reason Latin won't get you a job is that no one speaks it. If you write in Latin, no one can understand you. But Lisp is a computer language, and computers speak whatever language you, the programmer, tell them to. So if Lisp makes you a better programmer, like he says, why wouldn't you want to use it? If a painter were offered a brush that would make him a better painter, it seems to me that he would want to use it in all his paintings, wouldn't he? I'm not trying to make fun of Eric Raymond here. On the whole, his advice is good. What he says about Lisp is pretty much the conventional wisdom. But there is a contradiction in the conventional wisdom: Lisp will make you a better programmer, and yet you won't use it. Why not? Programming languages are just tools, after all. If Lisp really does yield better programs, you should use it. And if it doesn't, then who needs it? This is not just a theoretical question. Software is a very competitive business, prone to natural monopolies. A company that gets software written faster and better will, all other things being equal, put its competitors out of business. And when you're starting a startup, you feel this very keenly. Startups tend to be an all or nothing proposition. You either get rich, or you get nothing. In a startup, if you bet on the wrong technology, your competitors will crush you. Robert and I both knew Lisp well, and we couldn't see any reason not to trust our instincts and go with Lisp. We knew that everyone else was writing their software in C++ or Perl. But we also knew that that didn't mean anything. If you chose technology that way, you'd be running Windows. When you choose technology, you have to ignore what other people are doing, and consider only what will work the best. This is especially true in a startup. In a big company, you can do what all the other big companies are doing. But a startup can't do what all the other startups do. I don't think a lot of people realize this, even in startups. The average big company grows at about ten percent a year. So if you're running a big company and you do everything the way the average big company does it, you can expect to do as well as the average big company-- that is, to grow about ten percent a year. The same thing will happen if you're running a startup, of course. If you do everything the way the average startup does it, you should expect average performance. The problem here is, average performance means that you'll go out of business. The survival rate for startups is way less than fifty percent. So if you're running a startup, you had better be doing something odd. If not, you're in trouble. Back in 1995, we knew something that I don't think our competitors understood, and few understand even now: when you're writing software that only has to run on your own servers, you can use any language you want. When you're writing desktop software, there's a strong bias toward writing applications in the same language as the operating system. Ten years ago, writing applications meant writing applications in C. But with Web-based software, especially when you have the source code of both the language and the operating system, you can use whatever language you want. This new freedom is a double-edged sword, however. Now that you can use any language, you have to think about which one to use. Companies that try to pretend nothing has changed risk finding that their competitors do not. If you can use any language, which do you use? We chose Lisp. For one thing, it was obvious that rapid development would be important in this market. We were all starting from scratch, so a company that could get new features done before its competitors would have a big advantage. We knew Lisp was a really good language for writing software quickly, and server-based applications magnify the effect of rapid development, because you can release software the minute it's done. If other companies didn't want to use Lisp, so much the better. It might give us a technological edge, and we needed all the help we could get. When we started Viaweb, we had no experience in business. We didn't know anything about marketing, or hiring people, or raising money, or getting customers. Neither of us had ever even had what you would call a real job. The only thing we were good at was writing software. We hoped that would save us. Any advantage we could get in the software department, we would take. So you could say that using Lisp was an experiment. Our hypothesis was that if we wrote our software in Lisp, we'd be able to get features done faster than our competitors, and also to do things in our software that they couldn't do. And because Lisp was so high-level, we wouldn't need a big development team, so our costs would be lower. If this were so, we could offer a better product for less money, and still make a profit. We would end up getting all the users, and our competitors would get none, and eventually go out of business. That was what we hoped would happen, anyway. What were the results of this experiment? Somewhat surprisingly, it worked. We eventually had many competitors, on the order of twenty to thirty of them, but none of their software could compete with ours. We had a wysiwyg online store builder that ran on the server and yet felt like a desktop application. Our competitors had cgi scripts. And we were always far ahead of them in features. Sometimes, in desperation, competitors would try to introduce features that we didn't have. But with Lisp our development cycle was so fast that we could sometimes duplicate a new feature within a day or two of a competitor announcing it in a press release. By the time journalists covering the press release got round to calling us, we would have the new feature too. It must have seemed to our competitors that we had some kind of secret weapon-- that we were decoding their Enigma traffic or something. In fact we did have a secret weapon, but it was simpler than they realized. No one was leaking news of their features to us. We were just able to develop software faster than anyone thought possible. When I was about nine I happened to get hold of a copy of The Day of the Jackal, by Frederick Forsyth. The main character is an assassin who is hired to kill the president of France. The assassin has to get past the police to get up to an apartment that overlooks the president's route. He walks right by them, dressed up as an old man on crutches, and they never suspect him. Our secret weapon was similar. We wrote our software in a weird AI language, with a bizarre syntax full of parentheses. For years it had annoyed me to hear Lisp described that way. But now it worked to our advantage. In business, there is nothing more valuable than a technical advantage your competitors don't understand. In business, as in war, surprise is worth as much as force. And so, I'm a little embarrassed to say, I never said anything publicly about Lisp while we were working on Viaweb. We never mentioned it to the press, and if you searched for Lisp on our Web site, all you'd find were the titles of two books in my bio. This was no accident. A startup should give its competitors as little information as possible. If they didn't know what language our software was written in, or didn't care, I wanted to keep it that way.[2] The people who " \
  --generation_tokens 256 \
  --temperature 0.7 --top_p 0.95
```

Sample completions (since the model is 1B, it takes a few tries):

```

============================

--- Policy (sparse) continuation ---

1) didn't care were those who didn't know Lisp. We knew they would have a hard time understanding what our software was doing. And we didn't want to give them a technological advantage. 2) did care were those who did know Lisp. We wanted to make sure that even if they knew what our software was doing, they wouldn't be able to figure out how to do it themselves. And we didn't want to give them a technological advantage. I think this was a good strategy. It did not work in all cases, of course. There were a few people who did know Lisp, and they discovered it, and our competitors' software became unusable. But it didn't seem to make much difference to the bottom line. We were getting more sales than our competitors, and our competitors were getting fewer. As a startup, you have to find a way to make money without having to do anything. If you can get customers for free, you're doing well. But if you can't, you have to do something. And so, we were able to do this, and become a wildly successful startup. And we did it without ever having done anything like this before. We didn't do it with a big development team, or with marketing,

[chars: 1146]

--- Dense baseline continuation ---

1. Actually, it was more complicated than that. Lisp had been invented ten years earlier in a research lab in Cambridge, England. A few years later it was brought to the United States by a young programmer named John McCarthy. The town of Lexington, Massachusetts, had a few programmers who worked for the government. McCarthy happened to be one of them, and while he was there he wrote the first version of a language he called Lisp. He was trying to model a new kind of programming language called a functional language, which I briefly described in the previous chapter. McCarthy was a mathematician, and he thought he could use a language like his to simplify mathematical proofs. You might think it would be hard to write a computer program to do that. But in fact a lot of programming languages are mathematical languages. All programming languages are mathematical languages, after all. What was hard was to write a computer program that could do what a mathematician wanted it to do. Each mathematician has his own way of doing things, and it is not uncommon for him to want to do things in a way that seems bizarre to other mathematicians. It would be hard for McCarthy to model such a person's way of doing things. So he had to come up with a new kind of language

[chars: 1273]

Mean logprob under dense LM -> policy: -1.3906, dense: -1.5391
============================

Achieved Token-Keep-Rate        42.81250021304004%
Achieved Prune-Keep     100.0%
Achieved Quant-Ratio    1.0 (1 = full 16-bit)

============================
```