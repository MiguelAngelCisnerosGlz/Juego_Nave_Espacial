import pygame
import random
import math
from pygame import mixer

#inicializar pygame
pygame.init()

#crear pantalla
pantalla = pygame.display.set_mode((800,600))

#configuraciones del titulo e icono
pygame.display.set_caption("Invasión Espacial")
#cargar icono en variable para despuer ocuparle
icono = pygame.image.load("ovni.png")
pygame.display.set_icon(icono)
#Fondo de juego
fondo = pygame.image.load("Fondo.jpg")

#agregar musica de fondo
mixer.music.load('MusicaFondo.mp3')
mixer.music.set_volume(0.5)
mixer.music.play(-1)

#creando la nave del jugador (variables)
img_jugador = pygame.image.load("cohete.png")
jugador_x = 368
jugador_y = 500
jugador_x_cambio = 0

#creando enemigo (variables)
img_enemigo = []
enemigo_x = []
enemigo_y = []
enemigo_x_cambio = []
enemigo_y_cambio = []
cantidad_enemmigos = 10

for w in range(cantidad_enemmigos):
    img_enemigo.append(pygame.image.load("enemigo.png"))
    enemigo_x.append(random.randint(0,736))
    enemigo_y.append(random.randint(50,200))
    enemigo_x_cambio.append(0.1)
    enemigo_y_cambio.append(50)

#creando bala (variables)
img_bala = pygame.image.load("bala.png")
bala_x = 0
bala_y = 500
bala_x_cambio = 0
bala_y_cambio = 1
bala_visible = False

#puntaje
puntaje = 0
fuente = pygame.font.Font('freesansbold.ttf',32)
texto_x = 10
texto_y = 10

#Texto_game_over
fuente_final = pygame.font.Font('freesansbold.ttf',50)

#texto final funcion
def texto_final():
    mi_fuente_final = fuente_final.render("Juego Terminado",True,(255,255,255))
    pantalla.blit(mi_fuente_final,(100,200))

#Funcion del puntaje
def mostrar_puntaje(x , y):
    texto = fuente.render(f"Puntaje : {puntaje}" , True,(255,255,255))
    pantalla.blit(texto,(x,y))
#Funcion del jugador
def jugador(x , y):
    pantalla.blit(img_jugador,(x, y))

#Funcion enemigo
def enemigo(x , y,ene):
    pantalla.blit(img_enemigo[ene],(x,y))

#funcion de bala
def disparar_bala(x , y):
    global bala_visible
    bala_visible = True
    pantalla.blit(img_bala,(x + 16,y + 16))

#Funcion para coliciones
def hay_colision(x_1,y_1,x_2,y_2):
    distancia = math.sqrt(math.pow(x_1 - x_2,2) + math.pow(y_2 - y_1,2))
    if distancia < 27 :
        return True
    else:
        return False

#loop juego
se_ejecuta = True
while se_ejecuta:

    #Poniendo imagen de fondo
    pantalla.blit(fondo,(0,0))

    #Iterar eventos
    for evento in pygame.event.get():

        #Evento para cerrar programa
        if evento.type == pygame.QUIT:
            se_ejecuta = False

        #Evento si presionar teclas
        if evento.type == pygame. KEYDOWN:
            if evento.key == pygame.K_LEFT:
                jugador_x_cambio = -0.3
            if evento.key == pygame.K_RIGHT:
                jugador_x_cambio = 0.3
            if evento.key == pygame.K_SPACE:
                sonido_bala = mixer.Sound('Disparo.mp3')
                sonido_bala.play()
                if not bala_visible:
                    bala_x = jugador_x
                    disparar_bala(bala_x,bala_y)
        #Evento si soltaron las flechas
        if evento.type == pygame.KEYUP:
            if evento.key == pygame.K_LEFT or evento.key == pygame.K_RIGHT:
                jugador_x_cambio = 0

    #Modificar la pocision del jugador
    jugador_x += jugador_x_cambio

    #Mantener dentro de pantalla jugador
    if jugador_x <= 0:
        jugador_x = 0
    elif jugador_x >= 736:
        jugador_x = 736

    # Modificar la pocision del enemigo
    for e in range(cantidad_enemmigos):

        #fin del juego por enemigos abajo
        if enemigo_y[e] > 500:
            for k in range(cantidad_enemmigos):
                enemigo_y[k] = 1000
            texto_final()
            break

        enemigo_x[e] += enemigo_x_cambio[e]

        # Mantener dentro de pantalla enemigo
        if enemigo_x[e] <= 0:
                enemigo_x_cambio[e] = 0.2
                enemigo_y[e] += enemigo_y_cambio[e]
        elif enemigo_x[e] >= 736:
                enemigo_x_cambio[e] = -0.2
                enemigo_y[e] += enemigo_y_cambio[e]

        # Verificacion de la colision
        colision = hay_colision(enemigo_x[e], enemigo_y[e], bala_x, bala_y)
        if colision:
            sonido_colision = mixer.Sound('Golpe.mp3')
            sonido_colision.play()
            bala_y = 500
            bala_visible = False
            puntaje += 1
            enemigo_x[e] = random.randint(0, 736)
            enemigo_y[e] = random.randint(50, 200)

        enemigo(enemigo_x[e], enemigo_y[e],e)

    #Movimeinto bala
    if bala_y <= -64:
            bala_y = 500
            bala_visible = False

    if bala_visible:
        disparar_bala(bala_x,bala_y)
        bala_y -= bala_y_cambio

    #Invocacion de los objetos
    jugador(jugador_x,jugador_y)

    mostrar_puntaje(texto_x,texto_y)

    #Actualizar
    pygame.display.update()