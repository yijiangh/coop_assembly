(define (domain auto_scaffolds)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Element ?e)
    (ScaffoldElement ?se)
    (Conf ?r ?q)
    (Grasp ?e ?g)

    (RemoveMovement ?r ?t ?e ?g)

    (Assembled ?e)
    (Removed ?e)
    (Order ?e1 ?e2)
  )

  ; remove an element from the assembled structure
  (:action remove_element
    :parameters (?r ?t ?e ?g)
    :precondition (and
                       (RemoveMovement ?r ?t ?e ?g)
                       (Assembled ?e)
                    ;    (forall (?e2) (imply (Order ?e ?e2) (Removed ?e2)))
                       )
    :effect (and (Removed ?e)
                 (not (Assembled ?e))
                 )
  )
)
