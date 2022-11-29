from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from IPython.core.display import HTML, display


def artist_animation(fig, frames, **kwargs):
    anim = ArtistAnimation(fig, frames, **kwargs)
    plt.close()
    display(HTML(anim.to_jshtml()))


def animate(frames, figsize=None, **args):
    fig = plt.figure(figsize=figsize)
    artist_animation(fig, [[plt.imshow(t, **args)] for t in frames])
