---
document_id: adr-card-challenge.md
title: Tarjetas guardadas requieren paso extra de verificación
---

El `payments-vault` almacena tokens de tarjetas para compras recurrentes, pero no reutiliza una tarjeta guardada sin un paso extra de verificación. Aunque el buyer ya pagó antes con ese medio, cada intento de cobro con tarjeta almacenada exige challenge adicional — por ejemplo código enviado al titular o confirmación en app del emisor — según política de riesgo.

No se asume que “tarjeta conocida” equivalga a “pago aprobado”. El vault devuelve el token solo después del challenge exitoso; el `checkout-api` no salta ese paso por UX.

Esta decisión reduce fraude con tarjetas robadas y alinea el marketplace con exigencias de los procesadores de pago.
