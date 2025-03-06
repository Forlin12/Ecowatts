import cv2
import datetime
import numpy as np
from ultralytics import solutions
import matplotlib.pyplot as plt


# Classe ControleEsteira com nova saturação
class ControleEsteira:
    def __init__(self, max_ovos=25, velocidade_min_percent=20, velocidade_max=100):
        """
        Inicializa a malha de controle da esteira.

        :param max_ovos: Número máximo de ovos para saturação.
        :param velocidade_min_percent: Percentual mínimo da velocidade máxima (para 25+ ovos).
        :param velocidade_max: Velocidade máxima da esteira.
        """
        self.max_ovos = max_ovos
        self.velocidade_min = (velocidade_min_percent / 100) * velocidade_max  # Velocidade mínima em percentual
        self.velocidade_max = velocidade_max

    def calcular_velocidade(self, ovos_detectados):
        """
        Calcula a velocidade da esteira proporcional ao número de ovos detectados.
        - Para 25 ou mais ovos: 20% da velocidade máxima.
        - Para 10 ou menos ovos: 100% da velocidade máxima.
        - Para 24 a 11 ovos: aumenta proporcionalmente entre 20% e 100%.
        """
        if ovos_detectados >= self.max_ovos:
            return self.velocidade_min  # Saturação: 20% da velocidade máxima

        if ovos_detectados <= 10:
            return self.velocidade_max  # Saturação: 100% da velocidade máxima

        # Escala linear entre velocidade mínima e máxima para menos de 25 e mais de 10 ovos
        proporcao = (self.max_ovos - ovos_detectados) / (self.max_ovos - 10)
        velocidade = self.velocidade_min + proporcao * (self.velocidade_max - self.velocidade_min)
        return min(max(self.velocidade_min, velocidade), self.velocidade_max)  # Garantir dentro dos limites


# Configuração inicial do ControleEsteira
controle_esteira = ControleEsteira(max_ovos=25, velocidade_min_percent=20, velocidade_max=100)

# Variáveis para processar vídeo
kk = 0
sensor = np.zeros(10000, dtype=int)
velocidade_esteira = np.zeros(10000, dtype=float)

cap = cv2.VideoCapture(1)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(0, 250), (1080, 250)]  # For line counting

# Init Object Counter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model=r"C:\Users\Forlin\Ovoflow\resources\best.pt",  # Model path
    show_in=True,  # Display in counts
    show_out=True,  # Display out counts
    line_width=2,  # Adjust the line width for bounding boxes and text display
)

# Processamento do vídeo
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break


    rec_time = datetime.datetime.now().strftime("%d/%m/%Y - %H:%M:%S")
    # cv2.putText(im0, f"FPS: {cap.get(cv2.CAP_PROP_FPS)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(im0, f"WxH: {im0.shape[1]} x {im0.shape[0]}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    im0 = counter.count(im0)

    # Número de ovos detectados
    ovos_detectados = len(counter.track_ids)


    # Calcular velocidade da esteira com base no número de ovos detectados
    velocidade = controle_esteira.calcular_velocidade(ovos_detectados)

    # Adicionar informações no vídeo
    cv2.putText(im0, f"Ovos detectados: {ovos_detectados}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(im0, f"Velocidade Esteira: {velocidade:.2f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    print(f"Ovos detectados: {ovos_detectados}, Velocidade da Esteira: {velocidade:.2f}")

    kk += 1
    sensor[kk] = ovos_detectados
    velocidade_esteira[kk] = velocidade

    key = cv2.waitKey(30)
    if key == ord('q'):
        break

    if key == ord('p'):
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()

# Plotar os resultados
plt.figure(figsize=(10, 5))
plt.plot(sensor[:kk+1], label="Ovos Detectados")
plt.plot(velocidade_esteira[:kk+1], label="Velocidade da Esteira")
plt.xlabel('Frame')
plt.ylabel('Quantidade / Velocidade')
plt.title('Ovos Detectados e Velocidade da Esteira ao Longo do Tempo')
plt.legend()
plt.show()
