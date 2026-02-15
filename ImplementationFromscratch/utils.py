import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from typing import Callable, Optional, List, Any
import warnings
warnings.filterwarnings('ignore')

# entenda e mude
def create_animation_plot(
    history,
    plot_func: Callable,
    figsize: tuple = (10, 6),
    interval: int = 200,
    repeat: bool = True,
    repeat_delay: int = 1000,
    blit: bool = False,
    title_prefix: str = "Frame",
    show_slider: bool = True,
    **kwargs
) -> HTML:
    """
    Cria uma animação interativa com slider para visualizar dados.
    
    Parâmetros:
    -----------
    frames_data : List[Any]
        Lista com os dados para cada frame da animação
    plot_func : Callable
        Função que recebe (ax, frame_data, frame_number) e plota o frame
    figsize : tuple
        Tamanho da figura (largura, altura)
    interval : int
        Intervalo entre frames em milissegundos
    repeat : bool
        Se a animação deve repetir após terminar
    repeat_delay : int
        Delay antes de repetir em milissegundos
    blit : bool
        Se usa blitting para otimização
    title_prefix : str
        Prefixo para o título do frame
    show_slider : bool
        Se mostra o slider interativo (via to_jshtml)
    **kwargs : dict
        Argumentos adicionais para plot_func
        
    Retorna:
    --------
    HTML object para exibir no Jupyter
    """
    
    # Criar figura e eixos
    fig, ax = plt.subplots(figsize=figsize)
    
    def animate(frame_idx):
        ax.clear()
        frame_data = history['weights'][frame_idx]
        plot_func(ax, frame_data, **kwargs)
        ax.set_title(f'{title_prefix} {frame_idx + 1}/{len(history["weights"])}')
        ax.grid(True, alpha=0.3)
        return ax,
    
    # Criar animação
    anim = FuncAnimation(
        fig=fig,
        func=animate,
        frames=len(history['weights']),
        interval=interval,
        repeat=repeat,
        repeat_delay=repeat_delay,
        blit=blit
    )
    
    plt.close(fig)  # Evita mostrar figura duplicada
    
    # Gerar HTML com slider se solicitado
    # html = anim.to_jshtml()
    
    # return HTML(html)
    return anim


# entenda e arrume
def decision_boundary(ax, weights, X, y, cmap):
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    x1, x2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 100),
        np.linspace(x2_min, x2_max, 100)
    )
    X_grid = np.c_[x1.ravel(), x2.ravel()]
    y_hat = forward(weights, X_grid)
    y_hat = y_hat.reshape(x1.shape)
    
    ax.contourf(x1, x2, y_hat, levels=20, cmap=cmap, alpha=0.3)
    ax.contour(x1, x2, y_hat, levels=[0], colors='black', linewidths=2)
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap=cmap, 
        marker='X',
        s=150,
        edgecolors='black'
    )
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')