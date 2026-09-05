---
document_id: adr-shipping-step.md
title: Envío confirmado después del carrito
---

El costo y la dirección de entrega se confirman en el `shipping-service` en un paso posterior al cierre del carrito. El carrito puede mostrar estimaciones, pero no fija envío por defecto ni inventa un método si el buyer no eligió uno.

La opción *comprar ahora* acelera el camino hacia checkout, pero no saltea la confirmación de envío: sin dirección y método válidos en `shipping-service`, no hay orden. El `checkout-api` rechaza totales que incluyan envío no confirmado.

Separar carrito y envío evita cobrar un flete incorrecto y deja trazabilidad clara de qué eligió el buyer antes del pago.
