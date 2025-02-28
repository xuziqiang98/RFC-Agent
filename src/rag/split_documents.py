from typing import List, Optional, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document

def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    is_recursive: bool = True
) -> List[str]:
    """
    将文本拆分为较小的块。
    
    Args:
        text: 要拆分的文本内容
        chunk_size: 每个文本块的目标大小（字符数）
        chunk_overlap: 相邻文本块之间的重叠字符数
        separators: 用于拆分的分隔符列表，按优先级排序
        is_recursive: 是否使用递归拆分器
    
    Returns:
        拆分后的文本块列表
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    if is_recursive:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    else:
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separators[0] if separators else "\n"
        )
    
    return text_splitter.split_text(text)

def split_document(
    document: Document | dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    is_recursive: bool = True
) -> List[Document]:
    """
    将 LangChain Document 对象拆分为较小的文档块。
    
    Args:
        document: 要拆分的文档，可以是 LangChain Document 对象或包含 'page_content' 的字典
        chunk_size: 每个文本块的目标大小（字符数）
        chunk_overlap: 相邻文本块之间的重叠字符数
        separators: 用于拆分的分隔符列表，按优先级排序
        is_recursive: 是否使用递归拆分器
    
    Returns:
        拆分后的 Document 对象列表
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    # 处理输入可能是字典的情况
    if isinstance(document, dict):
        content = document.get("page_content", "")
        metadata = document.get("metadata", {})
        doc = Document(page_content=content, metadata=metadata)
    else:
        doc = document
    
    if is_recursive:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    else:
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separators[0] if separators else "\n"
        )
    
    return text_splitter.split_documents([doc])

def split_documents(
    documents: List[Document | dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None,
    is_recursive: bool = True
) -> List[Document]:
    """
    将多个文档拆分为较小的文档块。
    
    Args:
        documents: 要拆分的文档列表
        chunk_size: 每个文本块的目标大小（字符数）
        chunk_overlap: 相邻文本块之间的重叠字符数
        separators: 用于拆分的分隔符列表，按优先级排序
        is_recursive: 是否使用递归拆分器
    
    Returns:
        拆分后的 Document 对象列表
    """
    # 将所有输入转换为 Document 对象
    docs = []
    for doc in documents:
        if isinstance(doc, dict):
            content = doc.get("page_content", "")
            metadata = doc.get("metadata", {})
            docs.append(Document(page_content=content, metadata=metadata))
        else:
            docs.append(doc)
    
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    if is_recursive:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )
    else:
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separators[0] if separators else "\n"
        )
    
    return text_splitter.split_documents(docs)