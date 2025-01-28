import typer
from rich.console import Console
from rich.table import Table
from .model_manager import ModelManager
import torch

app = typer.Typer()
console = Console()
model_manager = ModelManager()

@app.command()
def add(model_id: str, display_name: str = None):
    """添加新模型"""
    try:
        model_manager.add_model(model_id, display_name)
        console.print(f"[green]成功添加模型: {model_id}[/green]")
    except Exception as e:
        console.print(f"[red]添加模型失败: {str(e)}[/red]")

@app.command()
def remove(model_id: str):
    """删除模型"""
    try:
        model_manager.remove_model(model_id)
        console.print(f"[green]成功删除模型: {model_id}[/green]")
    except Exception as e:
        console.print(f"[red]删除模型失败: {str(e)}[/red]")

@app.command()
def rename(model_id: str, new_name: str):
    """重命名模型"""
    try:
        model_manager.rename_model(model_id, new_name)
        console.print(f"[green]成功重命名模型: {model_id} -> {new_name}[/green]")
    except Exception as e:
        console.print(f"[red]重命名模型失败: {str(e)}[/red]")

@app.command()
def list():
    """列出所有模型"""
    models = model_manager.list_models()
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("模型ID")
    table.add_column("显示名称")
    table.add_column("状态")
    
    for model in models:
        status = "[green]已加载[/green]" if model["loaded"] else "[yellow]未加载[/yellow]"
        table.add_row(
            model["model_id"],
            model["display_name"],
            status
        )
    
    console.print(table)

@app.command()
def chat(model_id: str):
    """启动与模型的对话"""
    try:
        model, tokenizer = model_manager.load_model(model_id)
        console.print("[green]模型已加载，开始对话 (输入 'quit' 退出)[/green]")
        
        # 初始化对话历史
        history = []
        system_prompt = "You are a helpful AI assistant."
        
        while True:
            try:
                user_input = input("\n用户: ")
                if user_input.lower() == 'quit':
                    break
                
                # 构建输入格式
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                    system_prompt = None  # 只在第一次添加system prompt
                
                # 添加历史对话
                messages.extend(history)
                
                # 添加当前用户输入
                messages.append({"role": "user", "content": user_input})
                
                # 使用tokenizer的chat_template处理对话
                inputs = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=2048,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        attention_mask=torch.ones_like(inputs)  # 添加attention_mask
                    )
                
                response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
                
                # 清理输出格式
                response = response.strip()
                if response.startswith("assistant"):
                    response = response[len("assistant"):].strip()
                
                console.print(f"\n[blue]助手: {response}[/blue]")
                
                # 更新对话历史
                history.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": response}
                ])
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]生成回复失败: {str(e)}[/red]")
                continue
            
    except Exception as e:
        console.print(f"[red]对话失败: {str(e)}[/red]")
    finally:
        model_manager.unload_model(model_id)

if __name__ == "__main__":
    app() 