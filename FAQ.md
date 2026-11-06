# Frequently Asked Questions (FAQ)

Here are the answers to common technical questions regarding this project.

1. Why can't I just write "generate unique code" in a prompt?
A prompt changes the model's behavior only for a single run. Usually, this results in generating variations of the same template. 

This project uses Reinforcement Learning (GRPO), which changes the actual weights of the model. We penalize the model for copying the structures of other responses in the batch. Because of this, it is forced to change its policy and search for alternative algorithms (e.g., using recursion instead of a loop). A prompt does not change the weights, but RL does.


2. Why was the 4B model chosen? Isn't it too small?
The goal of the project was to test the diversity reward mechanism, not to create an ideal assistant. It was previously proven that reasoning training works on 3B models on consumer hardware. 

A 4B model is sufficient to show that a custom evaluator and structural penalties overcome the tendency to repeat solutions. Larger models require significantly more VRAM, which makes running on consumer GPUs impossible.


3. What about the random seed?
The seed is a parameter set in the training script (in our case, SEED = 42). It controls the initial weight initialization and data shuffling during training. 

The seed was fixed to prove that the increase in uniqueness is not random luck, but the result of 1000 steps of mathematical optimization.

---

# Часто задаваемые вопросы (FAQ)

Здесь собраны ответы на частые технические вопросы по проекту.

1. Почему нельзя просто написать в промпте "выдай уникальный код"?
Промпт меняет поведение модели только для одного запуска. Обычно это приводит к генерации вариаций одного и того же шаблона. 

В этом проекте используется обучение с подкреплением (GRPO), которое меняет реальные веса модели. Мы штрафуем модель за копирование структур других ответов в пакете. Из-за этого она вынуждена менять свою политику и искать альтернативные алгоритмы (например, использовать рекурсию вместо цикла). Промпт не меняет веса, а RL меняет.


2. Почему выбрана модель 4B? Разве она не слишком маленькая?
Цель проекта заключалась в проверке механизма награды за разнообразие, а не в создании идеального ассистента. Ранее было доказано, что обучение рассуждению проходит на моделях 3B на потребительском железе. 

Модели 4B достаточно, чтобы показать, что кастомный evaluator и структурные штрафы преодолевают склонность к повторению решений. Более крупные модели требуют значительно больше VRAM, что делает запуск на потребительских видеокартах невозможным.


3. А что насчет случайного сида?
Сид это параметр, который задается в скрипте обучения (в нашем случае SEED = 42). Он контролирует начальную инициализацию весов и перемешивание данных во время обучения. 

Сид был зафиксирован, чтобы доказать, что рост уникальности это не случайное везение, а результат 1000 шагов математической оптимизации.
