---
document_id: glossary.md
title: Glosario Marketplace Andes
---

**Buyer** — Persona que compra en Marketplace Andes. Tiene cuenta, historial de órdenes y medios de pago asociados. No confundir con vendedor ni con operador interno.

**Carrito** — Contenedor transitorio de ítems, promos aplicadas y precio cerrado. Vive en el `cart-service` y es prerequisito para reserva de stock y para avanzar a envío y pago.

**Comprar ahora** — Atajo que lleva al buyer hacia checkout con los ítems ya elegidos, sin pasar por una vista intermedia de carrito vacío. Saltea SOLO esa vista: no reemplaza el cierre de precio del `cart-service` ni la reserva de stock, que siguen siendo obligatorios. No elimina pasos obligatorios de envío ni de pago; solo acorta navegación. Un pedido que omita el cierre de precio o la reserva de stock es un choque real con `adr-cart-price.md` o `adr-stock-reserve.md`, no una simple navegación.

**Cyber Monday** — Ventana comercial de alto tráfico con flags, banners y reglas de promo especiales. Fuera de Cyber Monday, esos comportamientos deben estar apagados.

**Orden** — Compromiso firmado después de pago aprobado. Incluye totales del carrito, envío confirmado y líneas de ítems. El `checkout-api` emite la orden; servicios posteriores la ejecutan y registran entrega o devolución.
