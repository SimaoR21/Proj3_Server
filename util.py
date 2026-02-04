from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def put_unicode_text(img_bgr, text, position, font_size=20, color=(0, 255, 0), font_path=None):
    """
    Desenha texto com Unicode usando Pillow.
    - Cria sempre uma cópia para evitar erros de read-only (Gradio streaming)
    - Retorna a nova imagem modificada
    """
    # Cópia mutável obrigatória
    img = img_bgr.copy()

    # Converte para RGB + PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    # Fonte (ajusta para o teu sistema Linux em Porto)
    if font_path is None:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # ou DejaVuSans-Bold.ttf
        # Se não existir: sudo apt install fonts-dejavu

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"[Aviso] Fonte não encontrada: {e}. Usando default (pode não mostrar Unicode bem)")
        font = ImageFont.load_default()

    # Desenha texto (PIL usa RGB)
    draw.text(position, text, font=font, fill=color[::-1])  # BGR → RGB

    # Volta para OpenCV BGR
    result_rgb = np.array(pil_img)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    return result_bgr
