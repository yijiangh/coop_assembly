(define (domain construction)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Element ?e)

    (Assembled ?e)
    (Removed ?e)

    (PlaceAction ?r ?e ?t)
    (Grounded ?e)
    (Connected ?e)
    (Joined ?e1 ?e2)
    (Traj ?r ?t)

    (CollisionFree ?r ?t ?e)
    (UnSafeTraj ?r ?t)

    (Conf ?r ?q)
    (AtConf ?r ?q)
    (AtStart ?q ?t)
    (Assigned ?r ?e)
    (Order ?e1 ?e2)
    ; (Stiff)
  )

  ; place = remove the element
  (:action place
    :parameters (?r ?e ?t)
    :precondition (and
                       (PlaceAction ?r ?e ?t)
                       (Assembled ?e)
                       ; (Stiff)
                       (Connected ?e)

                       ; TODO: the following precondition will cause the following error:
                       ;  line 51, in compile_fluents_as_attachments
                       ;  if literal.predicate in predicate_map:
                       ;  AttributeError: 'UniversalCondition' object has no attribute 'predicate'
                    ;    (forall (?e2) (imply (Order ?e ?e2) (Removed ?e2)))

                       ;; uncomment if not using fluent
                    ;    (not (UnSafeTraj ?r ?t))
                       )
    :effect (and (Removed ?e)
                 (not (Assembled ?e))
                 )
  )

  (:derived (Connected ?e2)
   (or (Grounded ?e2)
       (exists (?e1) (and (Joined ?e1 ?e2)
                          (Assembled ?e1)
                          (Connected ?e1)
                     )
       )
   )
  )

  (:derived (UnSafeTraj ?r ?t)
   (exists (?e2) (and   (Element ?e2) (Traj ?r ?t) (Robot ?r)
                        (Assembled ?e2)
                        (not (CollisionFree ?r ?t ?e2))
                  ))
  )

)
