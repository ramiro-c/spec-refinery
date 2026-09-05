---
document_id: adr-cart-price.md
title: Precio final cerrado en el carrito
---

En Marketplace Andes, el precio que ve el buyer antes de pagar no se arma en el checkout ni en pantallas intermedias. El `cart-service` es el único lugar donde se cierra el total final: suma ítems, aplica promociones vigentes, incorpora el costo de envío cuando ya está definido y respeta la cantidad elegida. Ese cierre es obligatorio; no se saltea ni se delega a otro servicio.

El `pricing-engine` puede sugerir precios unitarios y reglas de descuento, pero no emite el total listo para cobrar. El `checkout-api` lee el total ya cerrado del carrito y lo usa como fuente de verdad para la confirmación de la orden. Si alguien intenta recalcular en checkout, la operación se rechaza y se pide refrescar el carrito.

Esta regla evita discrepancias entre lo que el buyer vio y lo que termina en la orden. Cualquier cambio de promo, envío o cantidad obliga a volver al `cart-service` para un nuevo cierre antes de avanzar.
