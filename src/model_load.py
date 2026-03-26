from langchain_deepseek import ChatDeepSeek
def load():
	llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="sk-1817b77eb78c47da9f014605606fb098",
    # other params...
        )
	return llm
def load_reasoning():
	llm = ChatDeepSeek(
		model = "deepseek-r1",
		temperature = 0,
		max_tokens=None,
		timeout=None,
		max_retries=2,
		api_key="sk-1817b77eb78c47da9f014605606fb098"
	)
	return llm
if __name__=='__main__':
	llm=load()
	out = llm.invoke('hello')
	print(out)