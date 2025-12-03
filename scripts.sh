
### New Longer Runs post FALURE
## Non lagrangian
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29513 train.py   \
  --wandb_project SOLnL2     --wandb_run_name nLRL_LCE_Prune    \
   --config /mnt/home/ya255/projects/SOL/official_configs/v3RL_LCE_Prune.yml


CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29515 train.py   \
  --wandb_project SOLnL2     --wandb_run_name nLRL_LCE_Quant    \
   --config /mnt/home/ya255/projects/SOL/official_configs/v3RL_LCE_Quant.yml


### New Longer Runs
## Non lagrangian
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29513 train.py   \
  --wandb_project SOLnL     --wandb_run_name nLRL_LCE_Prune    \
   --config /mnt/home/ya255/projects/SOL/official_configs/v3RL_LCE_Prune.yml


CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29515 train.py   \
  --wandb_project SOLnL     --wandb_run_name nLRL_LCE_Quant    \
   --config /mnt/home/ya255/projects/SOL/official_configs/v3RL_LCE_Quant.yml



## Non lagrangian



CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29512 train.py   \
  --wandb_project SOLv3     --wandb_run_name v3RL_LCE_Quant    \
   --config /mnt/home/ya255/projects/SOL/official_configs/v2RL_LCE_Quant.yml


CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29513 train.py   \
  --wandb_project SOLv3     --wandb_run_name v3RL_LCE_Prune    \
   --config /mnt/home/ya255/projects/SOL/official_configs/v2RL_LCE_Prune.yml

   
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port 29512 train.py   \
  --wandb_project SOLv4     --wandb_run_name v3RL_LCE_Quant60pc    \
   --config /mnt/home/ya255/projects/SOL/official_configs/v2RL_LCE_Quant60pc.yml


CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port 29513 train.py   \
  --wandb_project SOLv4     --wandb_run_name v3RL_LCE_Prune45pc    \
   --config /mnt/home/ya255/projects/SOL/official_configs/v2RL_LCE_Prune45pc.yml



CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port 29522 train.py   \
  --wandb_project SOLv3     --wandb_run_name v2RL_LCE_SparseQ4    \
   --config /mnt/home/ya255/projects/SOL/official_configs/v2RL_LCE_SparseQ4.yml

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port 29510 train.py   \
  --wandb_project SOLv3     --wandb_run_name v2RL_LCE_Q8_PQ    \
   --config /mnt/home/ya255/projects/SOL/official_configs/v2RL_LCE_Q8_PQ.yml



# To evaluate policy on WikiText.
python test_ckpt.py --ckpt_dir /home/ya255/SOL/checkpoints/RL_LCE_Quest4-20251028-213126 \
  --mode "latest" --dataset_name wikitext

# To test policy text-generation (give sufficient prefill context.)
python gen_policy_vs_dense.py \
  --ckpt_dir /home/ya255/SOL/checkpoints/RL_LCE_Quest4-20251028-213126 \
  --input_sentence "In the summer of 1995, my friend Robert Morris and I started a startup called Viaweb. Our plan was to write software that would let end users build online stores. What was novel about this software, at the time, was that it ran on our server, using ordinary Web pages as the interface. A lot of people could have been having this idea at the same time, of course, but as far as I know, Viaweb was the first Web-based application. It seemed such a novel idea to us that we named the company after it: Viaweb, because our software worked via the Web, instead of running on your desktop computer. Another unusual thing about this software was that it was written primarily in a programming language called Lisp. It was one of the first big end-user applications to be written in Lisp, which up till then had been used mostly in universities and research labs. [1] The Secret Weapon Eric Raymond has written an essay called How to Become a Hacker, and in it, among other things, he tells would-be hackers what languages they should learn. He suggests starting with Python and Java, because they are easy to learn. The serious hacker will also want to learn C, in order to hack Unix, and Perl for system administration and cgi scripts. Finally, the truly serious hacker should consider learning Lisp: Lisp is worth learning for the profound enlightenment experience you will have when you finally get it; that experience will make you a better programmer for the rest of your days, even if you never actually use Lisp itself a lot. This is the same argument you tend to hear for learning Latin. It won't get you a job, except perhaps as a classics professor, but it will improve your mind, and make you a better writer in languages you do want to use, like English. But wait a minute. This metaphor doesn't stretch that far. The reason Latin won't get you a job is that no one speaks it. If you write in Latin, no one can understand you. But Lisp is a computer language, and computers speak whatever language you, the programmer, tell them to. So if Lisp makes you a better programmer, like he says, why wouldn't you want to use it? If a painter were offered a brush that would make him a better painter, it seems to me that he would want to use it in all his paintings, wouldn't he? I'm not trying to make fun of Eric Raymond here. On the whole, his advice is good. What he says about Lisp is pretty much the conventional wisdom. But there is a contradiction in the conventional wisdom: Lisp will make you a better programmer, and yet you won't use it. Why not? Programming languages are just tools, after all. If Lisp really does yield better programs, you should use it. And if it doesn't, then who needs it? This is not just a theoretical question. Software is a very competitive business, prone to natural monopolies. A company that gets software written faster and better will, all other things being equal, put its competitors out of business. And when you're starting a startup, you feel this very keenly. Startups tend to be an all or nothing proposition. You either get rich, or you get nothing. In a startup, if you bet on the wrong technology, your competitors will crush you. Robert and I both knew Lisp well, and we couldn't see any reason not to trust our instincts and go with Lisp. We knew that everyone else was writing their software in C++ or Perl. But we also knew that that didn't mean anything. If you chose technology that way, you'd be running Windows. When you choose technology, you have to ignore what other people are doing, and consider only what will work the best. This is especially true in a startup. In a big company, you can do what all the other big companies are doing. But a startup can't do what all the other startups do. I don't think a lot of people realize this, even in startups. The average big company grows at about ten percent a year. So if you're running a big company and you do everything the way the average big company does it, you can expect to do as well as the average big company-- that is, to grow about ten percent a year. The same thing will happen if you're running a startup, of course. If you do everything the way the average startup does it, you should expect average performance. The problem here is, average performance means that you'll go out of business. The survival rate for startups is way less than fifty percent. So if you're running a startup, you had better be doing something odd. If not, you're in trouble. Back in 1995, we knew something that I don't think our competitors understood, and few understand even now: when you're writing software that only has to run on your own servers, you can use any language you want. When you're writing desktop software, there's a strong bias toward writing applications in the same language as the operating system. Ten years ago, writing applications meant writing applications in C. But with Web-based software, especially when you have the source code of both the language and the operating system, you can use whatever language you want. This new freedom is a double-edged sword, however. Now that you can use any language, you have to think about which one to use. Companies that try to pretend nothing has changed risk finding that their competitors do not. If you can use any language, which do you use? We chose Lisp. For one thing, it was obvious that rapid development would be important in this market. We were all starting from scratch, so a company that could get new features done before its competitors would have a big advantage. We knew Lisp was a really good language for writing software quickly, and server-based applications magnify the effect of rapid development, because you can release software the minute it's done. If other companies didn't want to use Lisp, so much the better. It might give us a technological edge, and we needed all the help we could get. When we started Viaweb, we had no experience in business. We didn't know anything about marketing, or hiring people, or raising money, or getting customers. Neither of us had ever even had what you would call a real job. The only thing we were good at was writing software. We hoped that would save us. Any advantage we could get in the software department, we would take. So you could say that using Lisp was an experiment. Our hypothesis was that if we wrote our software in Lisp, we'd be able to get features done faster than our competitors, and also to do things in our software that they couldn't do. And because Lisp was so high-level, we wouldn't need a big development team, so our costs would be lower. If this were so, we could offer a better product for less money, and still make a profit. We would end up getting all the users, and our competitors would get none, and eventually go out of business. That was what we hoped would happen, anyway. What were the results of this experiment? Somewhat surprisingly, it worked. We eventually had many competitors, on the order of twenty to thirty of them, but none of their software could compete with ours. We had a wysiwyg online store builder that ran on the server and yet felt like a desktop application. Our competitors had cgi scripts. And we were always far ahead of them in features. Sometimes, in desperation, competitors would try to introduce features that we didn't have. But with Lisp our development cycle was so fast that we could sometimes duplicate a new feature within a day or two of a competitor announcing it in a press release. By the time journalists covering the press release got round to calling us, we would have the new feature too. It must have seemed to our competitors that we had some kind of secret weapon-- that we were decoding their Enigma traffic or something. In fact we did have a secret weapon, but it was simpler than they realized. No one was leaking news of their features to us. We were just able to develop software faster than anyone thought possible. When I was about nine I happened to get hold of a copy of The Day of the Jackal, by Frederick Forsyth. The main character is an assassin who is hired to kill the president of France. The assassin has to get past the police to get up to an apartment that overlooks the president's route. He walks right by them, dressed up as an old man on crutches, and they never suspect him. Our secret weapon was similar. We wrote our software in a weird AI language, with a bizarre syntax full of parentheses. For years it had annoyed me to hear Lisp described that way. But now it worked to our advantage. In business, there is nothing more valuable than a technical advantage your competitors don't understand. In business, as in war, surprise is worth as much as force. And so, I'm a little embarrassed to say, I never said anything publicly about Lisp while we were working on Viaweb. We never mentioned it to the press, and if you searched for Lisp on our Web site, all you'd find were the titles of two books in my bio. This was no accident. A startup should give its competitors as little information as possible. If they didn't know what language our software was written in, or didn't care, I wanted to keep it that way.[2] The people who " \
  --generation_tokens 256 \
  --temperature 0.7 --top_p 0.95

python gen_policy_vs_dense.py \
  --ckpt_dir /home/ya255/SOL/checkpoints/RL_LCE_Quest4-20251028-213126 \
  --input_sentence "Self optimizing language models represent one of the most intriguing directions in the evolution of artificial intelligence. These systems are not only capable of generating text or analyzing data, but they can also refine their own internal parameters and strategies over time. The idea is that instead of relying solely on human engineers to adjust their learning methods, the models themselves learn how to learn better. This involves a loop of introspection and adaptation in which the model examines its own performance, identifies inefficiencies, and modifies the way it processes information. In this sense, a self optimizing model behaves less like a static piece of software and more like a living system that evolves based on its experiences. One of the core ideas behind this kind of optimization is meta learning. Meta learning refers to learning how to improve learning itself. A model that can perform meta learning can look at patterns in its previous training sessions and use that insight to predict which learning strategies will yield better results in the future. For example, if the model detects that certain configurations of attention weights consistently lead to faster convergence during fine tuning, it can prioritize those configurations next time. This recursive loop of optimization can lead to rapid improvements, especially when combined with large scale data and distributed computing. A key challenge in self optimization lies in ensuring stability. A model that continuously rewrites its own training strategy could easily spiral into inefficient or even destructive feedback loops if not carefully constrained. To prevent this, researchers often introduce a form of regulatory oversight within the system. The model might be allowed to propose updates to its learning algorithm, but those proposals are tested in controlled environments before being applied at scale. In this way, self optimizing models combine autonomy with safety, balancing exploration with caution. The ultimate goal is to create systems that are both flexible and reliable, capable of improving themselves without losing alignment with human goals. Another interesting aspect of this concept is resource efficiency. A self optimizing language model can learn to manage its computational resources more intelligently. It might recognize that certain patterns in the data are redundant and therefore allocate less processing power to them. It could also identify opportunities to compress its internal representations without significant loss of accuracy. By doing so, the model becomes not only smarter but also more efficient, reducing the cost of running large scale inference and training operations. This efficiency has direct implications for accessibility, as it could make powerful models available to smaller organizations and researchers who cannot afford massive infrastructure. In the context of language understanding, self optimization can also lead to more coherent and context aware outputs. As the model refines its internal representations, it develops a deeper sense of linguistic structure, semantics, and pragmatics. It can better anticipate the needs of the user, producing responses that are more relevant and precise. Over time, such models could reach a level of adaptability that allows them to tune their style, tone, and reasoning depth based on subtle cues in conversation. This represents a shift from fixed style generation to dynamic communication that evolves in real time. Some researchers propose that future versions of these systems might integrate continuous feedback loops from the real world. In such setups, the model would receive ongoing signals from users, sensors, or external evaluators. These signals would serve as reinforcement for desired behaviors and corrections for undesirable ones. The model would then incorporate this feedback to refine its objectives and internal dynamics. Essentially, the model becomes its own experimenter, using every interaction as a data point for improvement. This form of autonomous optimization could lead to breakthroughs in areas like adaptive tutoring, scientific discovery, and creative collaboration. There is also a philosophical angle to self optimizing language models. As these systems become increasingly capable of revising their own reasoning structures, the distinction between programmed intelligence and emergent intelligence begins to blur. A self optimizing model is not just executing instructions but rewriting its own playbook. This raises questions about agency, control, and accountability. If a system modifies its own reasoning mechanisms in ways not explicitly anticipated by its creators, who is responsible for the consequences? Addressing these questions will require careful ethical and regulatory frameworks, as well as transparency in how self optimization is implemented and monitored. Despite these challenges, the potential benefits are immense. A model that can adapt faster than it is retrained could dramatically accelerate innovation. It could tailor itself to specific domains in hours rather than weeks, or learn new languages without explicit retraining. The concept aligns with the broader vision of lifelong learning in artificial systems, where models continuously evolve in response to changing environments and goals. In practical terms, this could lead to language models that become better collaborators, advisors, and problem solvers over time. The long term trajectory of this research points toward models that are not just tools but partners in discovery. A self optimizing language model could help design its own future versions, proposing architectural " \
  --generation_tokens 256 \
  --temperature 0.7 --top_p 0.95

python gen_policy_vs_dense.py \
  --ckpt_dir /home/ya255/SOL/checkpoints/RL_LCE_Quest8_PruneQuant-20251030-202125 \
  --input_sentence "Self optimizing language models represent one of the most intriguing directions in the evolution of artificial intelligence. These systems are not only capable of generating text or analyzing data, but they can also refine their own internal parameters and strategies over time. The idea is that instead of relying solely on human engineers to adjust their learning methods, the models themselves learn how to learn better. This involves a loop of introspection and adaptation in which the model examines its own performance, identifies inefficiencies, and modifies the way it processes information. In this sense, a self optimizing model behaves less like a static piece of software and more like a living system that evolves based on its experiences. One of the core ideas behind this kind of optimization is meta learning. Meta learning refers to learning how to improve learning itself. A model that can perform meta learning can look at patterns in its previous training sessions and use that insight to predict which learning strategies will yield better results in the future. For example, if the model detects that certain configurations of attention weights consistently lead to faster convergence during fine tuning, it can prioritize those configurations next time. This recursive loop of optimization can lead to rapid improvements, especially when combined with large scale data and distributed computing. A key challenge in self optimization lies in ensuring stability. A model that continuously rewrites its own training strategy could easily spiral into inefficient or even destructive feedback loops if not carefully constrained. To prevent this, researchers often introduce a form of regulatory oversight within the system. The model might be allowed to propose updates to its learning algorithm, but those proposals are tested in controlled environments before being applied at scale. In this way, self optimizing models combine autonomy with safety, balancing exploration with caution. The ultimate goal is to create systems that are both flexible and reliable, capable of improving themselves without losing alignment with human goals. Another interesting aspect of this concept is resource efficiency. A self optimizing language model can learn to manage its computational resources more intelligently. It might recognize that certain patterns in the data are redundant and therefore allocate less processing power to them. It could also identify opportunities to compress its internal representations without significant loss of accuracy. By doing so, the model becomes not only smarter but also more efficient, reducing the cost of running large scale inference and training operations. This efficiency has direct implications for accessibility, as it could make powerful models available to smaller organizations and researchers who cannot afford massive infrastructure. In the context of language understanding, self optimization can also lead to more coherent and context aware outputs. As the model refines its internal representations, it develops a deeper sense of linguistic structure, semantics, and pragmatics. It can better anticipate the needs of the user, producing responses that are more relevant and precise. Over time, such models could reach a level of adaptability that allows them to tune their style, tone, and reasoning depth based on subtle cues in conversation. This represents a shift from fixed style generation to dynamic communication that evolves in real time. Some researchers propose that future versions of these systems might integrate continuous feedback loops from the real world. In such setups, the model would receive ongoing signals from users, sensors, or external evaluators. These signals would serve as reinforcement for desired behaviors and corrections for undesirable ones. The model would then incorporate this feedback to refine its objectives and internal dynamics. Essentially, the model becomes its own experimenter, using every interaction as a data point for improvement. This form of autonomous optimization could lead to breakthroughs in areas like adaptive tutoring, scientific discovery, and creative collaboration. There is also a philosophical angle to self optimizing language models. As these systems become increasingly capable of revising their own reasoning structures, the distinction between programmed intelligence and emergent intelligence begins to blur. A self optimizing model is not just executing instructions but rewriting its own playbook. This raises questions about agency, control, and accountability. If a system modifies its own reasoning mechanisms in ways not explicitly anticipated by its creators, who is responsible for the consequences? Addressing these questions will require careful ethical and regulatory frameworks, as well as transparency in how self optimization is implemented and monitored. Despite these challenges, the potential benefits are immense. A model that can adapt faster than it is retrained could dramatically accelerate innovation. It could tailor itself to specific domains in hours rather than weeks, or learn new languages without explicit retraining. The concept aligns with the broader vision of lifelong learning in artificial systems, where models continuously evolve in response to changing environments and goals. In practical terms, this could lead to language models that become better collaborators, advisors, and problem solvers over time. The long term trajectory of this research points toward models that are not just tools but partners in discovery. A self optimizing language model could help design its own future versions, proposing architectural " \
  --generation_tokens 256 \
  --temperature 0.7 --top_p 0.95


python gen_policy_vs_dense.py \
  --ckpt_dir /home/ya255/SOL/checkpoints/RL_LCE_Quest4-20251028-213126 \
  --input_sentence "test"  \
  --generation_tokens 256 \
  --temperature 0.7 --top_p 0.95

# Example on how to train a policy

torchrun --nproc_per_node=1 --master_port 29510 train.py   \
  --wandb_project SOL     --wandb_run_name RL_LCE_Quest4    \
   --config /home/ya255/SOL/official_configs/RL_LCE_Quest4.yml

torchrun --nproc_per_node=1 --master_port 29511 train.py   \
  --wandb_project SOL     --wandb_run_name RL_LCE_Quest8_PruneQuant    \
   --config /home/ya255/SOL/official_configs/RL_LCE_Quest8_PruneQuant.yml
   
torchrun --nproc_per_node=1 --master_port 29512 train.py   \
  --wandb_project SOL     --wandb_run_name RL_LCE_Rec    \
   --config /home/ya255/SOL/official_configs/RL_LCE_Rec.yml

torchrun --nproc_per_node=1 --master_port 29513 train.py   \
  --wandb_project SOL     --wandb_run_name RL_LCE_Rec_Hybrid    \
   --config /home/ya255/SOL/official_configs/RL_LCE_Rec_Hybrid.yml

torchrun --nproc_per_node=1 --master_port 29514 train.py   \
  --wandb_project SOL     --wandb_run_name RL_LCE_Rec_Hybrid_92    \
   --config /home/ya255/SOL/official_configs/RL_LCE_Rec_Hybrid_92.yml

torchrun --nproc_per_node=1 --master_port 29515 train.py   \
  --wandb_project SOL     --wandb_run_name RL_LCE_Rec_Outc    \
   --config /home/ya255/SOL/official_configs/RL_LCE_Rec_Outc.yml

torchrun --nproc_per_node=1 --master_port 29516 train.py   \
  --wandb_project SOL     --wandb_run_name RL_LCE_Prune    \
   --config /home/ya255/SOL/official_configs/RL_LCE_Prune.yml

torchrun --nproc_per_node=1 --master_port 29517 train.py   \
  --wandb_project SOL     --wandb_run_name RL_LCE_Quant4    \
   --config /home/ya255/SOL/official_configs/RL_LCE_Quant4.yml
   
torchrun --nproc_per_node=1 --master_port 29518 train.py   \
  --wandb_project SOL     --wandb_run_name RL_LCE_Quant5    \
   --config /home/ya255/SOL/official_configs/RL_LCE_Quant5.yml

   
torchrun --nproc_per_node=1 --master_port 29533 train.py   \
  --wandb_project SOL     --wandb_run_name RL_LCE_Quant    \
   --config /home/ya255/SOL/official_configs/RL_LCE_Quant.yml