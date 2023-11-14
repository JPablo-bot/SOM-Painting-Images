from skimage import io, data, color
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import skimage as sk
#==================================================================#

nfil = 15
ncol = 15
nras = 3    # Características

# datos = io.imread('paletaAzul.jpg')
# datos = io.imread('donGato1.jpg')
datos = io.imread('vanGogh2.jpg')
plt.figure(1)
plt.imshow(datos)

n_total = datos.shape[0]*datos.shape[1]
datos = datos.reshape(n_total, 3) # input

som = np.random.rand(nfil, ncol, nras) 
plt.ion()
plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(som)


x = np.linspace(0, ncol, ncol)
y = np.linspace(0, nfil, nfil)
x, y = np.meshgrid(x, y)
epocas = 50
alpha0 = 0.5
decay = 0.05
sgm0 = 20


for t in range(epocas):
    alpha = alpha0 * np.exp(-t * decay)
    sgm = sgm0 * np.exp(-t * decay)
    ven = np.ceil(sgm*3)

    for i in range(n_total):
        vector = datos[i, :] 
        columna = som.reshape(nfil*ncol, 3) 
        d = 0
        # Cálculo de distancia en RGB
        for n in range(3):
            d = d + (vector[n]-columna[:, n])**2
        vec_dists = np.sqrt(d)
        
        # Indice para activar la neurona con menor distancia
        ind = np.argmin(vec_dists)
        bmfil, bmcol = np.unravel_index(ind, [nfil, ncol])
        # Inhibir al resto
        g = np.exp( -( ( (x-bmcol)**2) + ((y-bmfil)**2) ) / (2*sgm*sgm) )
        # Radios de acción
        ffil = int( np.max( [0, bmfil-ven] ) )
        tfil = int( np.min( [bmfil+ven, nfil] ) )
        fcol = int( np.max( [0, bmcol-ven] ) )
        tcol = int( np.min( [bmcol+ven, ncol] ) )
        # Modificar SOM
        vecindad = som[ffil:tfil, fcol:tcol, :]
        a, b, c = vecindad.shape
        T = np.ones(vecindad.shape)
        T[:,:,0] = T[:,:,0] * vector[0]
        T[:,:,1] = T[:,:,1] * vector[1]
        T[:,:,2] = T[:,:,2] * vector[2]
        
        G = np.ones(vecindad.shape)
        G[:,:,0] = g[ffil:tfil, fcol:tcol]
        G[:,:,1] = g[ffil:tfil, fcol:tcol]
        G[:,:,2] = g[ffil:tfil, fcol:tcol]
        
        # Ecuación de aprendizaje de SOM (regla de aprendizaje de Kohonen)
        vecindad = vecindad + (alpha*G*(T-vecindad))
        
        # Actualización de som según el vecindario | Genera el universo de colores nuevo
        som[ffil:tfil, fcol:tcol, :] = vecindad

        # if (i%1000==0):
        # # Líneas para observar el proceso de forma dinámica
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(np.uint8(som))
        #     plt.pause(0.05)
        #     plt.show()
        
plt.subplot(1, 2, 2)
plt.imshow(np.uint8(som))
plt.show()


#------------------------------------- Testing --------------------------------
datosTest = io.imread('parque.jpg')
paletaColores = som.reshape(nfil*ncol, 3) 
newIma = np.zeros((datosTest.shape[0],datosTest.shape[1],3))
for i in range(datosTest.shape[0]):
    for j in range(datosTest.shape[1]):
        pixel = datosTest[i,j,:] 
        dist = 0
        # Cálculo de distancia en RGB
        for n in range(3):
            dist = dist + (pixel[n] - paletaColores[:, n])**2
        vectorDists = np.sqrt(dist)
        ind = np.argmin(vectorDists)
        for k in range(3):
            newIma[i,j,k] = paletaColores[ind,k]

plt.figure(3)
plt.subplot(1, 2, 1)
plt.imshow(np.uint8(datosTest))
plt.subplot(1, 2, 2)
plt.imshow(np.uint8(newIma))

        
        
