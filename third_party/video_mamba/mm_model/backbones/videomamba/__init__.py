from .videomamba import build_videomamba

def build_text_clip(clip_teacher):
    model = eval(clip_teacher)()
    return model