This is an experiment to create a fine-tuned GPT-2 model for personalized study generation.

The python files provided would need to be copy-pasted into a new project, and the libraries in the files need to be installed.
I am working on creating a repository with all of the files already provided. However, this is 15+ GB, so this may not happen anytime soon. 
For now, you can create a project in your desired IDE (I used Intellij as the IDE) and download the libraries that error in the python files.

train.py has a small dataset of math problems as an example to help it learn,
but it could very easily be adjusted to include new data. train.py focuses on
training GPT-2's pretrained data. Notably, each run of train.py overwrites the previous
data in the "gpt2-math-generator" folder to create a "fresh" start with each run of train.py.
This can be avoided by simply deleting the lines 3 lines below the "Delete old folder so we start from scratch"
comment.

test.py can be used to test the effectiveness of the new GPT-2 model after training it.
Simply run the file, and it will output GPT-2's result using the new data
(in addition to its pretrained data).

Notably, for short questions like "Q: What is 2 + 2? A:", it will likely produce many
answers like "A: 4. B: ...", due to the nature of GPT-2's pretraining forcing it to
answer a question as completely as possible. This may be avoided by reducing
"max_new_tokens" in test.py but this may reduce the characters visible for the actual answer.
With such results, "A:" corresponds to its most likely answer, "B:" second most likely,
and so on.

Important: GPT-2, especially a small version like the one being used,
is unlikely to solve complex math accurately. It can reproduce patterns it saw,
but GPT-2 is a LLM; not designed to solve math. In addition, the small dataset provided
does not make a significant difference to this. Performance can be improved by increasing
the epochs in "num_train_epochs" (training_args), but this is minimal, as the value is
already quite high relative to the small dataset. A completely different pretrained model
would be necessary to completely avoid this, but those usually aren't free and for the purposes
of this experiment are not necessary.
