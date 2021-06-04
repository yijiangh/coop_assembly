(define (domain stiffness)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Element ?e)
    (Conf ?r ?q)
    (Grasp ?e ?g)

    (AddMovement ?r ?t ?e ?g)
    ; (RemoveMovement ?r ?t ?e ?g)

    (Assembled ?e)
    (Removed ?e)
    (Order ?e1 ?e2)
  )

  ; add an element to the assembled structure
  (:action add_element
    :parameters (?r ?t ?e ?g)
    :precondition (and
                       (AddMovement ?r ?t ?e ?g)
                       (Removed ?e)
                       )
    :effect (and (Assembled ?e)
                 (not (Removed ?e))
                 ;(increase (total-cost) (Stiffness)) ; TODO(caelan): fluent cost function
                 )
  )

)
