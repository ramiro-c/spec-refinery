---
document_id: adr-promo-at-cart.md
title: Promociones válidas solo si se aplicaron en el carrito
---

El `promotions-service` publica campañas, cupones y descuentos automáticos, pero una promo no vale por existir en catálogo: tiene que quedar aplicada en el `cart-service` antes del cierre de precio. Si el buyer llega al checkout con una promo que nunca se registró en el carrito, el `checkout-api` no la inventa ni la revalida en caliente.

El flujo correcto es: el buyer agrega ítems, el carrito consulta al `promotions-service`, evalúa elegibilidad y persiste la promo aplicada junto con el snapshot de precio. Solo entonces el total cerrado incluye ese beneficio. Promos que expiraron entre pantallas se descartan en el carrito, no en pago.

Esta decisión mantiene una sola fuente de verdad para descuentos y evita sorpresas al confirmar la orden.
