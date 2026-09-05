---
document_id: spec-checkout-current.md
title: Flujo de checkout actual
---

El checkout vigente en Marketplace Andes sigue cuatro etapas secuenciales. Ninguna se saltea salvo atajos de navegación que no eliminan validaciones.

**Carrito** — El buyer agrega ítems; el `cart-service` reserva stock con `inventory-service`, consulta `promotions-service` y cierra el precio final. Sin cierre exitoso no hay siguiente paso.

**Envío** — En `shipping-service` el buyer confirma dirección y método. El costo incorporado al total debe coincidir con lo acordado aquí; el carrito puede refrescarse si cambia el flete elegido.

**Pago** — El `checkout-api` presenta el total cerrado. `payments-vault` procesa el medio elegido; tarjetas guardadas exigen challenge. Fallo de pago no crea orden.

**Confirmación** — Pago aprobado: el `checkout-api` emite la orden y muestra resumen al buyer. A partir de ahí operan logística, entrega y eventualmente `returns-service`.

Este flujo es la línea base fuera de Cyber; durante Cyber pueden activarse flags que modifiquen UI o tiempos, pero la secuencia lógica se mantiene.
