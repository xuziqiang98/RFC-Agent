import inspect
from types import MethodType, FunctionType
from pathlib import Path
import random
from langchain.schema import Document

def get_script_name() -> str:
    """获取调用此函数的脚本文件名（不含扩展名）。

    该函数使用 Python 的检查栈（inspection stack）来确定调用此函数的脚本。
    它获取调用者的文件名，并返回该路径的主干部分（不含扩展名的文件名）。

    Returns:
        str: 调用此函数的脚本文件名（不含扩展名）。
    """
    caller_frame_record = inspect.stack()[1]
    module_path = caller_frame_record.filename
    return Path(module_path).stem

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