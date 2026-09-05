---
document_id: adr-cyber-flags.md
title: Cambios de Cyber controlados por feature flags
---

Durante Cyber, Marketplace Andes activa comportamientos especiales en checkout, banners y reglas de promo mediante feature flags. Cualquier cambio de flujo o de UI en esas fechas debe pasar por flag; no se despliega código permanente que altere el checkout fuera del período acordado.

Al terminar el evento, los flags de Cyber se apagan de forma explícita. No quedan prendidos por omisión: un flag de Cyber activo fuera de ventana se considera incidente y se revierte. El `checkout-api` y los servicios de front consumen los mismos nombres de flag para no divergir.

Esta regla protege al buyer del resto del año y permite ensayar el flujo intensivo solo cuando el calendario lo autoriza.
