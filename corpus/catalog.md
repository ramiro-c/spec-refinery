---
document_id: catalog.md
title: Catálogo de servicios Marketplace Andes
---

Referencia de servicios backend que componen el dominio de compra en Marketplace Andes. Cada línea resume responsabilidad principal.

- **cart-service** — Carrito, cierre de precio final y promos aplicadas.
- **checkout-api** — Orquestación de checkout y emisión de órdenes con total ya cerrado.
- **pricing-engine** — Precios unitarios y reglas de descuento; no cierra el total de la orden.
- **promotions-service** — Campañas y cupones; la validez en compra depende del carrito.
- **payments-vault** — Tokenización y cobro con tarjetas guardadas y challenge extra.
- **shipping-service** — Dirección, métodos y costo de envío confirmados post-carrito.
- **inventory-service** — Stock disponible y reservas ligadas al carrito activo.
- **returns-service** — Devoluciones y plazos contados desde la entrega.
