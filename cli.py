import click
import asyncio
from pathlib import Path
from src.chains.rfc_chain import RFCChain
from src.models.embeddings.factory import EmbeddingFactory
from src.env import Env
from src.configs.common_configs import PathConfig

def create_embedding_model():
    """创建嵌入模型实例"""
    env = Env()
    return EmbeddingFactory.create(
        model_type=getattr(env, "EMBEDDING_MODEL_TYPE", "openai"),
        model_name=getattr(env, "EMBEDDING_MODEL_NAME", "text-embedding-ada-002"),
        api_base=getattr(env, "EMBEDDING_MODEL_BASE", "https://api.openai.com/v1"),
        api_key=getattr(env, "EMBEDDING_API_KEY", "")
    )

@click.group()
def cli():
    """RFC Agent CLI - RFC文档处理和问答工具"""
    pass

@cli.command()
@click.option('--chunk-size', default=500, help='文档切块大小')
@click.option('--chunk-overlap', default=100, help='文档切块重叠大小')
@click.option('--rfc-path', default=PathConfig().rfcs, type=click.Path(exists=True), help='RFC文档路径')
def process(chunk_size, chunk_overlap, rfc_path):
    """处理RFC文档：获取、切块、向量化和存储"""
    try:
        # 创建嵌入模型
        embedding_model = create_embedding_model()
        
        # 创建RFCChain实例
        rfc_chain = RFCChain(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            rfc_docs_path=rfc_path if rfc_path else None
        )
        
        # 执行处理流程
        asyncio.run(rfc_chain.process())
        
    except Exception as e:
        click.echo(f"处理过程中出错: {str(e)}", err=True)
        raise click.Abort()

if __name__ == '__main__':
    cli()