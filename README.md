# Building Task-Oriented Conversational Agents in Low-Resource Settings

![project] ![research]



- <b>Project Lead(s) / Mentor(s)</b>
    1. Dr. Uthayasanker Thayasivam
    2. Prof. Sanath Jayasena
- <b>Contributor(s)</b>
    1. Tharindu Madusanka
    2. Durashi Langappuli
    3. Thisara Welmilla

---

## Summary

Implementing a task-oriented conversational agent in low resource settings addressing the overfitting issue arise when optimizing dialogue policy using Reinforcement Learning (RL) and utilizing pipeline ensembling in Natural Language Understanding (NLU).

## Description

We introduce a comprehensive task-oriented conversational agent in low resource settings utilizing a novel pipeline ensemble technique to enhance natural language understanding tasks. The experiments conducted shows that our pipeline ensembling approach outperforms individual pipelines in precision, recall, f1-score and accuracy in both intent classification and entity extraction tasks. Furthermore, we implemented a Reinforcement Learning based dialogue policy learner addressing the overfitting issue by proposing a novel approach for synthetic agenda generation by acknowledging the underlying probability distribution of the user agendas with a reward-based sampling method that prioritizes failed dialogue acts

### Setting up



### Overall Achitecture 

The overall achitecture consists of 5 components,

    1. Data extractor : Create seperate dataset to train individual components
    2. Natural Language Understanding (NLU) component : Identify intent andextract entities from the user utterance
    3. Dialogue State Tracker(DST) module : Track the state of the dialogue flowby the given NLU information and past dialogue history.
    4. Dialogue Policy Learner(POL) module : Decides the next best action totake based on the given dialogue state
    5. Natural Language Generator(NLG) module : Convert the given dialogueact to a naturally understandable agent utterance

During the training process the data extractor create seperate datasets and to train individual components and each of the individual components trained parallelly.

<img src = docs/Images/Overall_architecture.png height = "500px" >

### Natural Language Understanding

To improve the performance of NLU we implemented a novel pipeline ensemble technique that combine outputs of various different pipelines at the inference. Diagram below shows the achitecture of the NLU component

<img src = docs/Images/NLU_architecture.png width = "700px" >

### Overall Achitecture 

To adress the overfitting issue arise when optimizing dialogue policy using RL we introduce 2 techniques

    1. Probability based self-play approach
    2. Reward based sampling technique 

Diagram below shows the overall achitecture of the dialogue policy optimization component. The probability calculation and agenda synthesis component is facilitate probability based self-play approach while reward update component and probability calculation facilitate reward based sampling technique.

<img src = docs/Images/architecture_diagram.png height = "400px" >

## More references

We have used the user simulator described in [A User Simulator for Task-Completion Dialogues](http://arxiv.org/abs/1612.05688) as the simulator. Github link to the user simulator can be found on [here](https://github.com/MiuLab/TC-Bot).

Main papers to be cited


```
@inproceedings{Tharindu2020Dialog,
  title={Dialog policy optimization for low resource setting using Self-play and Reward based Sampling},
  author={Tharindu Madusanka, Durashi Langappuli, Thisara Welmilla Uthayasanker Thayasivam and Sanath Jayasena},
  booktitle={34th Pacific Asia Conference on Language, Information and Computation},
  year={2020}
}
```

---

### License

Apache License 2.0

### Code of Conduct

Please read our [code of conduct document here](https://github.com/aaivu/aaivu-introduction/blob/master/docs/code_of_conduct.md).

[project]: https://img.shields.io/badge/-Project-blue
[research]: https://img.shields.io/badge/-Research-yellowgreen
