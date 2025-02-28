from typing import List, Optional, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
import sqlite3
from hashlib import sha256
from src.configs.common_configs import PathConfig
import os
import time
from pathlib import Path

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

def get_doc_from_path(folder: str) -> List[Document]:
    """从指定文件夹获取所有.txt文件并创建Document对象列表。
    
    Args:
        folder: 要扫描的文件夹路径
        
    Returns:
        List[Document]: Document对象列表，每个对象包含文件内容和元数据
    """
    documents = []
    folder_path = Path(folder)
    
    # 获取所有.txt文件
    for txt_file in folder_path.glob("*.txt"):
        try:
            # 读取文件内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 获取文件名（不含后缀）作为source
            source = txt_file.stem
            
            # 创建Document对象
            doc = Document(
                page_content=content,
                metadata={"source": source}
            )
            documents.append(doc)
            
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {e}")
            continue
    
    return documents

def init_database(db_path: str):
    """初始化数据库，创建必要的表。"""
    # path_config = PathConfig()
    # db_path = path_config.dbs / "metadata.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT UNIQUE,
                  file_hash TEXT,
                  chunk_count INTEGER,
                  last_modified REAL,
                  processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS chunks
                 (chunk_id TEXT PRIMARY KEY,
                  doc_id INTEGER,
                  chunk_text TEXT,
                  vector BLOB,
                  FOREIGN KEY(doc_id) REFERENCES documents(id))''')
    conn.commit()
    conn.close()

def get_file_signature(file_path: str) -> dict:
    """获取文件的唯一标识信息。

    计算文件的哈希值、获取最后修改时间等信息，用于判断文件是否发生变化。

    Args:
        file_path: 文件的完整路径。

    Returns:
        dict: 包含文件信息的字典，具有以下键：
            - filename (str): 文件名（不含路径）
            - hash (str): 文件内容的SHA256哈希值
            - last_modified (float): 文件最后修改时间的时间戳

    Raises:
        FileNotFoundError: 如果文件不存在
        PermissionError: 如果没有读取文件的权限
    """
    file_stat = os.stat(file_path)
    content_hash = sha256(Path(file_path).read_bytes()).hexdigest()
    return {
        'filename': os.path.basename(file_path),
        'hash': content_hash,
        'last_modified': file_stat.st_mtime
    }

def needs_processing(db_path: str, file_path: str) -> bool:
    """判断文件是否需要重新处理。

    通过比较文件的哈希值和最后修改时间，判断文件是否发生变化或是新文件。

    Args:
        file_path: 文件的完整路径。

    Returns:
        bool: 如果文件需要处理返回True，否则返回False。
            以下情况返回True：
            - 文件在数据库中不存在（新文件）
            - 文件的哈希值与数据库中记录的不同
            - 文件的最后修改时间与数据库中记录的不同

    Raises:
        sqlite3.Error: 数据库操作出错时
        FileNotFoundError: 如果文件不存在
        PermissionError: 如果没有读取文件的权限
    """
    file_sig = get_file_signature(file_path)
    
    # path_config = PathConfig()
    # db_path = path_config.dbs / "metadata.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''SELECT file_hash, last_modified 
                      FROM documents 
                      WHERE filename = ?''', (file_sig['filename'],))
    
    existing = cursor.fetchone()
    conn.close()
    
    if not existing:
        return True  # 新文件
    
    saved_hash, saved_mtime = existing
    return not (saved_hash == file_sig['hash'] 
                and saved_mtime == file_sig['last_modified'])