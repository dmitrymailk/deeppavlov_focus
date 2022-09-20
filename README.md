### Распаковать zip архив
```bash
unzip -o -d ./datasets/FoCus.zip ./datasets/
```

### Subtasks
#### Subtask 1
- **Goal**: Predicting the proper persona sentences and knowledge 
- **Input**: Persona candidates (5 sentences), Knowledge candidates (10 paragraphs), document on the topic, and dialog history
- **Output**: Index of answer persona sentences, Index of answer knowledge
- **Evaluation**: Accuracy
#### Subtask 2
- **Goal**: Generating the next agent response in natural language using persona and knowledge
- **Input**: Persona sentences, document on the topic, and dialog history
- **Output**: Agent utterance
- **Evaluation**: CharF++, BLEU, ROUGE-L