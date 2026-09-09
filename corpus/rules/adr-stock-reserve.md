---
document_id: adr-stock-reserve.md
title: Reserva de stock al entrar al carrito
---

El `inventory-service` reserva unidades cuando un ítem entra al carrito del buyer, no cuando confirma la compra ni cuando navega el catálogo. Sin carrito activo no hay reserva: mirar un producto no bloquea stock para otros.

La reserva tiene TTL acorde a la sesión; si el carrito expira o el buyer abandona, el stock vuelve al pool disponible. El checkout no crea reservas nuevas: solo consume las que el carrito ya registró con `inventory-service`.

Esta regla equilibra disponibilidad real durante Cyber Monday y evita overselling en el último click.
