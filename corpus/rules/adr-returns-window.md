---
document_id: adr-returns-window.md
title: Plazo de devolución desde la entrega
---

El `returns-service` calcula el plazo para solicitar devolución desde la fecha de entrega confirmada al buyer, no desde el click de compra ni desde el pago. Hasta que el envío no figure como entregado, el contador no arranca.

Las políticas por categoría pueden acortar o extender días hábiles, pero siempre ancladas al evento de entrega registrado por logística. Un buyer que compró hace un mes pero recibió ayer tiene el plazo completo desde ayer.

Esta decisión alinea expectativas con la posesión real del producto y reduce disputas por demoras de envío.
