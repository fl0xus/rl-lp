from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
import os

class BaseExpert(object):

    def __init__(self, name, description, model):
        self.name = name
        self.description = description
        self.model = model

        self.llm = ChatOpenAI(
            model_name=model,
            temperature=0,
            openai_api_base="$INFERENCE_ENDPOINT/v1",
            openai_api_key="test",
        )
        self.forward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.FORWARD_TASK
        self.forward_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(self.forward_prompt_template)
        )
        if hasattr(self, 'BACKWARD_TASK'):
            self.backward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.BACKWARD_TASK
            self.backward_chain = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template(self.backward_prompt_template)
            )

    def forward(self):
        pass

    def backward(self):
        pass

    def __str__(self):
        return f'{self.name}: {self.description}'
