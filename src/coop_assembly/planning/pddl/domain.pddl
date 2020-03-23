(define (domain construction)
  (:requirements :strips :equality)
  (:predicates
    (Element ?e)
    (Printed ?e)
    (Removed ?e)
    (PrintAction ?e ?t)
    (Grounded ?e)
    (Connected ?e)
    (Joined ?e1 ?e2)
    (Traj ?t)
    ; (Order ?e1 ?e2)
  )

  ;;;; removing the element
  (:action print
    :parameters (?e ?t)
    :precondition (and (PrintAction ?e ?t)
                       (Printed ?e)
                       ; Caelan use partial ordering to enforce connectivity
                       ; (forall (?e2) (imply (Order ?e ?e2) (Removed ?e2)))
                       (forall (?e2) (imply (Element ?e2) (or (Connected ?e2) (Removed ?e2)) ) )
                  )
    :effect (and (Removed ?e)
                 (not (Printed ?e)))
  )

;   (:derived (Supported ?e2) ; Single support
;    (and (Element ?e2) (Printed ?e2)
;        (or (and (Grounded ?e2))
;            (exists (?e1) (and (Supports ?e1 ?e2) (Supported ?e1)))))
;   )
  ;(:derived (Supported ?n) ; All support
  ;  (and (Node ?n)
  ;       ; TODO: bug in focused algorithm (preimage fact not achievable)
  ;       (forall (?e) (imply (Supports ?e ?n) (Printed ?e))))
  ;)

  (:derived (Connected ?e2)
   (or (Grounded ?e2)
       (exists (?e1) (and (Joined ?e1 ?e2)
                          (Printed ?e1)
                          (Connected ?e1)
                     )
       )
   )
  )
)
