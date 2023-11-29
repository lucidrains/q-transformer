<img src="./q-transformer.png" width="450px"></img>

## Q-transformer (wip)

Implementation of <a href="https://qtransformer.github.io/">Q-Transformer</a>, Scalable Offline Reinforcement Learning via Autoregressive Q-Functions, out of Google Deepmind

I will be keeping around the logic for Q-learning on single action just for final comparison with the proposed autoregressive Q-learning on multiple actions. Also to serve as education for myself and the public.

## Appreciation

- <a href="https://stability.ai/">StabilityAI</a>, <a href="https://a16z.com/supporting-the-open-source-ai-community/">A16Z Open Source AI Grant Program</a>, and <a href="https://huggingface.co/">🤗 Huggingface</a> for the generous sponsorships, as well as my other sponsors, for affording me the independence to open source current artificial intelligence research

## Todo

- [x] first work way towards single action support
- [x] offer batchnorm-less variant of maxvit, as done in SOTA weather model metnet3
- [x] add optional deep dueling architecture
- [x] add n-step Q learning
- [x] build the conservative regularization
- [x] build out main proposal in paper (autoregressive discrete actions until last action, reward given only on last)
- [x] improvise decoder head variant, instead of concatenating previous actions at the frames + learned tokens stage. in other words, use classic encoder - decoder
    - [ ] allow for cross attention to fine frame / learned tokens

- [ ] build out a simple dataset creator class, taking in the environment as an iterator / generator
- [ ] see if the main idea in this paper is applicable to language models <a href="https://github.com/lucidrains/llama-qrlhf">here</a>
- [ ] consult some RL experts and figure out if there are any new headways into resolving <a href="https://www.cs.toronto.edu/~cebly/Papers/CONQUR_ICML_2020_camera_ready.pdf">delusional bias</a>
- [ ] redo maxvit with axial rotary embeddings + sigmoid gating for attending to nothing. enable flash attention for maxvit with this change

## Citations

```bibtex
@inproceedings{qtransformer,
    title   = {Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions},
    authors = {Yevgen Chebotar and Quan Vuong and Alex Irpan and Karol Hausman and Fei Xia and Yao Lu and Aviral Kumar and Tianhe Yu and Alexander Herzog and Karl Pertsch and Keerthana Gopalakrishnan and Julian Ibarz and Ofir Nachum and Sumedh Sontakke and Grecia Salazar and Huong T Tran and Jodilyn Peralta and Clayton Tan and Deeksha Manjunath and Jaspiar Singht and Brianna Zitkovich and Tomas Jackson and Kanishka Rao and Chelsea Finn and Sergey Levine},
    booktitle = {7th Annual Conference on Robot Learning},
    year   = {2023}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```
