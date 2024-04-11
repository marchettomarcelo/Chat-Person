import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "langchain/prompts";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// ask for user input
const readline = require("readline");

const rl = readline.createInterface({
	input: process.stdin,
	output: process.stdout,
});


await rl.question("Faça uma pergunta sobre a Desa: ", async function (pergunta: string) {
	const loader = new PDFLoader("./desa.pdf", { splitPages: true });

	const docs = await loader.load();

	const textSplitter = new RecursiveCharacterTextSplitter({
		chunkSize: 100,
		chunkOverlap: 20,
	});

	const docs2 = await textSplitter.splitDocuments(docs);

	const embeddings = new OpenAIEmbeddings({
		batchSize: 512, // Default value if omitted is 512. Max is 2048
		model: "text-embedding-3-large",
	});

	// save the embeddings to a file
	// await embeddings. ("embeddings.json");

	const memIndex = await MemoryVectorStore.fromDocuments(docs2, embeddings);


	const llm = new ChatOpenAI({ model: "gpt-4-0125-preview", temperature: 0 });

    const retriever = memIndex.asRetriever();
    
	const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");
	

	const ragChain = await createStuffDocumentsChain({
		llm,
		prompt,
		// outputParser: new StringOutputParser(),
	});

	const retrievedDocs = await retriever.getRelevantDocuments(pergunta);

    const response = await ragChain.invoke({
		question: pergunta,
		context: retrievedDocs,
	});

    console.log(response);

	rl.close();
});


