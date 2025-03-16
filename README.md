
#  AI Workshop 

Welcome to the **AI Workshop** repository! This workshop is designed to take you from foundational knowledge to building and deploying **real-world AI projects**. Whether you're a beginner or have some experience, this workshop will provide practical insights, hands-on experience, and a roadmap to advance your AI journey.

---

Certainly! Here’s a step-by-step explanation and a concise summary rewritten for your README.md file:

---
## lesson 1 

This session provides a comprehensive introduction to Artificial Intelligence (AI), covering its core concepts, components, and its various branches. The session also dives into the fundamentals of Generative AI (GenAI), a rapidly evolving area within the field. We will explore how AI systems learn, how machine learning models are trained, and how different learning modes, such as supervised, unsupervised, and reinforcement learning, function. Additionally, the session provides insights into the working mechanisms of Transformers, a key technology used in Large Language Models (LLMs), and how they are trained to perform language-based tasks.

### **Summary of Key Concepts:**

- **Artificial Intelligence (AI)**: AI refers to the simulation of human intelligence processes by machines, particularly computer systems. It includes learning, reasoning, and self-correction.
  
- **Generative AI (GenAI)**: This subfield of AI focuses on algorithms capable of generating new content such as images, text, or audio. GenAI leverages advanced machine learning techniques like deep learning to create outputs that mimic human-like creativity.

- **Training and Learning Modes**: AI systems learn by processing data, and there are different learning paradigms:
  - **Supervised Learning**: The model learns from labeled data, where the correct output is provided for each input.
  - **Unsupervised Learning**: The model identifies patterns in data without labeled outputs.
  - **Reinforcement Learning**: The model learns by interacting with its environment and receiving feedback in the form of rewards or penalties.

- **Transformers and LLMs**: Transformers are a type of deep learning model used extensively in natural language processing (NLP). They are designed to handle sequential data efficiently, and their attention mechanism allows them to focus on relevant parts of input data. This architecture is the backbone of Large Language Models (LLMs) such as GPT-3, which excel in tasks like language translation, summarization, and content generation.

---

This structure provides a clear, organized way to present the session content in a **README.md** file.

---



## lesson 2
**[LLM lab ](./lesson_1/lesson_1.ipynb)**  

---



This Lab  demonstrates how to leverage **LangChain** and **Groq LLM APIs** for building interactive, intelligent applications using Python. It covers key concepts such as initializing language models, prompt engineering, structured data extraction, memory management, and chain creation for advanced AI interactions.

---

## Key Features

### 1. **LLM Initialization**
- Load environment variables securely with `dotenv`.
- Set up Groq's language models (e.g., `gemma2-9b-it`, `qwen-2.5-32b`) via `ChatGroq`.
- Configure temperature and API keys for controlled outputs.

---

### 2. **Prompt Engineering**
- Send user inputs using `HumanMessage`.
- Use `ChatPromptTemplate` for dynamic message generation.
- Example: Translate angry customer reviews into **polite German** with company name integration.

---

### 3. **Structured Data Extraction**
- Extract key information from unstructured emails (e.g., trip itineraries).
- Define a **Response Schema** and use `StructuredOutputParser` to format output as JSON.
- Parse the model’s output into a Python dictionary for further use.

---

### 4. **Conversation Memory**
- Implement `ConversationBufferMemory` to retain context across multiple interactions.
- Example: Remember user's name during a conversation and recall it when asked.

---

### 5. **LangChain Chains**
- Build pipelines using LangChain's chaining capabilities.
- Combine prompts, LLM, and output parsers for specific tasks.
- Example: Generate a **lullaby** based on a character and location in under 90 words.

---

## Technologies Used
- **Python**
- **LangChain**
- **Groq API**
- **dotenv**
- **Markdown for Output Display**

---

## Example Use Cases
- Multilingual customer support.
- Intelligent email parsing and summarization.
- Context-aware chatbots.
- Creative content generation (e.g., children's stories).

---

## lesson 3
**[Agent lab ](./Labs/Agent.ipynb)** 

---

# AI Agent with Tool Integration and ReAct Framework

This Lab demonstrates the implementation of intelligent agents using **LangChain**, **Groq LLM**, and **ReAct framework**. It showcases how to create multi-functional AI systems that can reason, perform actions (e.g., math or search), and maintain memory for contextual understanding.

---

## Features

###  Agent Initialization
- Load environment variables securely.
- Initialize Groq’s large language model (e.g., `gemma2-9b-it`).

---

###  Tool Integration
- Integrate tools like `llm-math` for computation tasks.
- Custom tool creation for general logic and query handling via the language model.

---

###  ReAct Framework
- Step-by-step reasoning:
  1. **Thought**: Internal analysis.
  2. **Action**: Execute a tool (e.g., calculator).
  3. **Observation**: Process the tool's result.
  4. **Final Answer**: Output after complete reasoning.
- Helps the agent solve complex tasks logically and efficiently.

---

###  Wikipedia Search Agent
- Search and lookup tools based on Wikipedia data.
- Agent performs dynamic querying, exploration, and summarization of knowledge.

---

###  Conversational Memory
- Maintain chat history with `ConversationBufferMemory`.
- Enhance agent's interaction by remembering prior exchanges and context.

---

## Use Cases
- Math problem solving.
- Real-time knowledge retrieval (Wikipedia).
- General query handling with contextual memory.
- Complex reasoning and chained computations.

---

## Technologies Used
- **Python**, **LangChain**, **Groq API**
- **ReAct Reasoning Model**
- **Wikipedia Docstore Tools**
- **IPython Markdown Display**

---


## lesson 4
**[RAG lab ](./Labs/RAG.ipynb)** 
---

# Text Vectorization & Retrieval-Augmented Generation (RAG)

This project demonstrates **text vectorization**, **semantic search**, and a simplified **Retrieval-Augmented Generation (RAG)** pipeline using OpenAI models.

---

## Key Features

### 🔹 Text Embeddings
- Use `text-embedding-ada-002` to convert text into **high-dimensional vectors** capturing semantic relationships.

### 🔹 Visualization
- **PCA** reduces vector dimensions for **2D/3D visualization**.
- Interactive plots created with `matplotlib` and `mplcursors`.

### 🔹 Similarity Search
- Compute **cosine similarity** between vectors to measure semantic proximity.

### 🔹 RAG Workflow
1. Embed user query.
2. Perform vector search over embedded data.
3. Retrieve top relevant texts.
4. Combine query + context.
5. Generate response using **GPT-3.5-turbo**.

---

## Tools Used
- **OpenAI API** (Embeddings + GPT-3.5-turbo)
- **scikit-learn** (Cosine similarity, PCA)
- **Matplotlib + mplcursors** (Interactive visualization)
- **LangChain** (Embedding model wrapper)

---

## lesson 5
**[Projects lab ](./Labs/pojects.ipynb)** 

---
# AI-Powered Applications Suite

This projects includes **three powerful AI applications** leveraging Groq LLM, LangChain, and OpenAI APIs for intelligent automation in daily tasks.

---

## 1.  AI Recipe Generator **[ Link](https://github.com/IbrahimAlobid2/AI-Recipe-Generator)**  
  
Generate **healthy and quick recipes** using ingredient images.

- **Image to Ingredients**: Analyze food images to detect ingredients.
- **LLM Recipe Creation**: Generate personalized recipes, cooking steps, tools, and macros.
- **Best For**: Fitness, meal planning, quick nutrition.

---

## 2.  AI Invoice Extractor **[ Link](https://github.com/IbrahimAlobid2/AI-Invoice-Extractor)**  
  
Extract **structured data** from PDF invoices automatically.

- **Text Extraction**: Read invoice text using `PyPDF2`.
- **Data Structuring**: LLM extracts fields like Invoice ID, Amount, Terms, etc.
- **DataFrame Output**: Store results for analysis in **pandas DataFrame**.

---

## 3.  AI Tech Newsletter Generator **[ Link](https://github.com/IbrahimAlobid2/AI-Tech-Newslette)**  
Create **automated news summaries** from real-time tech articles.

- **Search Integration**: Fetch top articles using **Tavily API**.
- **LLM Filtering**: Select top-quality sources.
- **Vector Storage**: Store and retrieve content via FAISS.
- **Newsletter Output**: Summarize content into concise, engaging updates.

---

## Technologies Used
- **Groq LLM** (LangChain Integration)
- **OpenAI Embeddings**
- **FAISS** (Vector Search)
- **Streamlit UI**
- **PyPDF2, Pandas, JSON**

---


## lesson 6
---

##  AI Roadmap
From Zero to Hero in Artificial Intelligence & Machine Learning

###  Stage 1: Foundation — Mathematics & Python  
**Goal:** Build a strong base in math and programming.

- **Mathematics for Machine Learning**  
  Topics: Linear Algebra, Calculus, Probability, Statistics  
  [Coursera Course](https://www.coursera.org/specializations/mathematics-for-machine-learning-and-data-science)

- **Python Programming for Data Science**  
  Topics: Python Basics, Data Structures, NumPy, Pandas  
  [Python Fundamentals](https://www.coursera.org/learn/packt-python-fundamentals-and-data-science-essentials-trjtx)  
  OR  
  [Python for MLOps](https://www.coursera.org/learn/python-mlops-duke)

---

###  Stage 2: Core Machine Learning (ML)  
**Goal:** Understand and apply key ML algorithms.

- **Machine Learning Introduction**  
  Topics: Regression, Decision Trees, SVM, Clustering  
  [Coursera Course](https://www.coursera.org/programs/syrian-youth-assembly-learning-program-blb40/specializations/machine-learning-introduction?source=search)

---

###  Stage 3: Deep Learning Specialization  
**Goal:** Master neural networks and deep learning techniques.

- **Deep Learning Specialization**  
  Topics: CNN, RNN, LSTM, Optimization  
  [Coursera Specialization](https://www.coursera.org/specializations/deep-learning)

- **TensorFlow in Practice**  
  Topics: Model Building, Training, Deployment  
  [Coursera Certificate](https://www.coursera.org/professional-certificates/tensorflow-in-practice)

- **Neural Networks: Zero to Hero**  
  Hands-on neural network implementation  
  [YouTube Playlist](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=GQpYCD-FUjz-ZNU4)

---

###  Stage 4: LLMs & LangChain  
**Goal:** Build LLM-powered applications for real use cases.

- **LangChain for LLM Application Development**  
  [Coursera Project](https://www.coursera.org/projects/langchain-for-llm-application-development-project)

- **LangChain: Chat With Your Data**  
  [Coursera Project](https://www.coursera.org/projects/langchain-chat-with-your-data-project)

- **LLM Zoomcamp (Advanced)**  
  Topics: Custom LLMs, Fine-tuning, Deployment  
  [YouTube Playlist](https://www.youtube.com/playlist?list=PL3MmuxUbc_hIB4fSqLy_0AfTjVLpgjV3R)

---

###  Stage 5: API Development with FastAPI  
**Goal:** Deploy models as APIs/web apps.

- **FastAPI for AI Deployment**  
  Topics: REST APIs, Model Serving  
  [YouTube Playlist](https://youtube.com/playlist?list=PL-2EBeDYMIbQghmnb865lpdmYyWU3I5F1&si=pCZ6pQGQS9p6Q3LA)

---

###  Stage 6: Capstone Projects  
**Goal:** Apply skills in real-world projects.

#### Suggested Projects
- Predictive Model for Real-World Data (ML)
- Image Classifier using CNN (DL)
- AI Chatbot using LangChain & LLMs
- Model Deployment with FastAPI 

---

##  Timeframe
Estimated Completion: **6 to 12 Months**  
*Flexible based on your learning pace.*

---

##  Pro Tips for Success
- Focus on **hands-on projects** after each stage.
- Join **AI communities** (Kaggle, GitHub, LinkedIn).
- Document all your projects in a **GitHub portfolio**.
- Stay **curious, consistent, and collaborative**.

---

##  Final Learning Path Overview
1. Math + Python  
2. Machine Learning  
3. Deep Learning + TensorFlow  
4. LLMs + LangChain  
5. FastAPI Deployment  
6. Real Projects → Portfolio → Career Opportunities

---


# Ibrahim Alobaid












