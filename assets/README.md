# assets

Recursos gráficos de PediaFlow-AI.

| Archivo             | Uso                                                         |
|---------------------|-------------------------------------------------------------|
| `logo.png`          | Logo con fondo transparente (867 × 619 px). Lo usa la app.  |
| `logo_original.jpg` | Versión original generada por IA, con fondo azul. Respaldo. |

La app busca `assets/logo.*` sin importar mayúsculas ni extensión y detecta el
tipo real de imagen por su contenido. Si el archivo no existe, muestra un
ícono genérico.

Para regenerar `logo.png` a partir del original se quitó el degradado de fondo
por software (modelo radial del fondo + alfa por distancia de color).
