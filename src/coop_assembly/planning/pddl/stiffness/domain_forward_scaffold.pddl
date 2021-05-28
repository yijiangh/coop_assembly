(define (domain stiffness)
  (:requirements :strips :equality)
  (:predicates
    (Robot ?r)
    (Element ?e)
    (ScaffoldElement ?se)
    (Conf ?r ?q)
    (Grasp ?e ?g)
    (ScaffoldGrasp ?se ?g)

    (AddMovement ?r ?t ?e ?g)
    (AddScaffoldMovement ?r ?t ?se ?g)
    (RemoveScaffoldMovement ?r ?t ?se ?g)
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
                 )
  )

  ; add a scaffold element to the assembled structure
  (:action add_scaffold_element
    :parameters (?r ?t ?se ?g)
    :precondition (and
                       (AddScaffoldMovement ?r ?t ?se ?g)
                       (Removed ?se)
                       )
    :effect (and (Assembled ?se)
                 (not (Removed ?se))
                 )
  )

  ; remove a scaffold element from the assembled structure
  (:action remove_scaffold_element
    :parameters (?r ?t ?se ?g)
    :precondition (and
                       (RemoveScaffoldMovement ?r ?t ?se ?g)
                       (Removed ?se)
                       )
    :effect (and (Assembled ?se)
                 (not (Removed ?se))
                 )
  )

)
